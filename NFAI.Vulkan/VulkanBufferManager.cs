using NFAI.Core;
using Silk.NET.Vulkan;
using System.Runtime.CompilerServices;
using Buffer = Silk.NET.Vulkan.Buffer;

namespace NFAI.Vulkan;

public class VulkanBufferManager : IDisposable
{
    private readonly Vk vk;
    private readonly Device device;
    private readonly PhysicalDevice physicalDevice;
    private readonly Queue computeQueue;
    private readonly uint computeQueueFamilyIndex;

    private CommandPool commandPool;
    //private readonly List<(Buffer buffer, DeviceMemory memory)> allocatedResources = [];

    public VulkanBufferManager(Vk vk, Device device, PhysicalDevice physicalDevice)
    {
        this.vk = vk;
        this.device = device;
        this.physicalDevice = physicalDevice;

        // Find compute queue family
        computeQueueFamilyIndex = FindComputeQueueFamily(physicalDevice);
        vk.GetDeviceQueue(device, computeQueueFamilyIndex, 0, out computeQueue);

        // Create a command pool for compute
        CreateCommandPool();
    }

    /// <summary>
    /// A command pool for the compute queue.
    /// </summary>
    public CommandPool CommandPool => commandPool;

    /// <summary>
    /// Creates a Vulkan buffer and allocates memory for it.
    /// </summary>
    public unsafe void CreateBuffer<T>(
        ulong elementCount,
        BufferUsageFlags usage,
        MemoryPropertyFlags properties,
        out Buffer buffer,
        out DeviceMemory memory)
        where T : struct
    {
        ulong size = elementCount * (ulong)Unsafe.SizeOf<T>();

        // 1. Create buffer
        BufferCreateInfo bufferCreateInfo = new()
        {
            SType = StructureType.BufferCreateInfo,
            Size = size,
            Usage = usage,
            SharingMode = SharingMode.Exclusive
        };

        if (vk.CreateBuffer(device, &bufferCreateInfo, null, out buffer) != Result.Success)
        {
            throw new Exception("Failed to create Vulkan buffer.");
        }

        // 2. Get memory requirements
        MemoryRequirements memRequirements;
        vk.GetBufferMemoryRequirements(device, buffer, &memRequirements);

        // 3. Allocate memory
        MemoryAllocateInfo allocInfo = new()
        {
            SType = StructureType.MemoryAllocateInfo,
            AllocationSize = memRequirements.Size,
            MemoryTypeIndex = FindMemoryType(memRequirements.MemoryTypeBits, properties)
        };

        if (vk.AllocateMemory(device, &allocInfo, null, out memory) != Result.Success)
        {
            throw new Exception("Failed to allocate Vulkan buffer memory.");
        }

        // 4. Bind buffer to memory
        if (vk.BindBufferMemory(device, buffer, memory, 0) != Result.Success)
        {
            throw new Exception("Failed to bind buffer memory.");
        }
    }

    unsafe public void DestoryBuffer(ref Buffer buffer, ref DeviceMemory memory)
    {
        if (buffer.Handle != 0)
        {
            vk.DestroyBuffer(device, buffer, null);
            buffer = default;
        }

        if (memory.Handle != 0)
        {
            vk.FreeMemory(device, memory, null);
            memory = default;
        }
    }

    public unsafe void UploadDataToDeviceLocal<T>(Buffer deviceLocalBuffer, ComputeCollection<T> data)
        where T : struct
    {
        // 1. Create staging buffer
        CreateBuffer<T>(
            data.Length,
            BufferUsageFlags.TransferSrcBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out var stagingBuffer,
            out var stagingMemory);

        // 2. Upload data to staging buffer
        UploadData(ref stagingMemory, data); // ← your existing method

        // 3. Copy staging buffer to device-local buffer
        CopyBuffer(ref stagingBuffer, ref deviceLocalBuffer, data.Length * (ulong)Unsafe.SizeOf<T>());

        // 4. Cleanup
        vk.DestroyBuffer(device, stagingBuffer, null);
        vk.FreeMemory(device, stagingMemory, null);
    }

    /// <summary>
    /// Uploads data to a Vulkan buffer (host-visible).
    /// </summary>
    public void UploadData<T>(ref DeviceMemory memory, ComputeCollection<T> data) where T : struct
    {
        ulong dataSize = data.Length * (ulong)Unsafe.SizeOf<T>();
        unsafe
        {
            void* mappedMemory = null;

            // 1. Map memory
            var res = vk.MapMemory(device, memory, 0, dataSize, 0, &mappedMemory);
            if (res != Result.Success)
            {
                throw new Exception("Failed to map Vulkan memory.");
            }

            try
            {
                // 2. Copy data as a batch rather than byte-by-byte
                byte* dst = (byte*)mappedMemory;
                ulong index = 0;
                
                // Create a temporary array to batch process data
                T[] tempBuffer = new T[1024]; // Process in chunks
                int count = 0;
                
                foreach (var item in data.GetData())
                {
                    tempBuffer[count++] = item;
                    
                    // When buffer is full or this is the last item, copy to GPU memory
                    if (count == tempBuffer.Length || index + (ulong)count >= data.Length)
                    {
                        fixed (T* srcPtr = tempBuffer)
                        {
                            ulong bytesToCopy = (ulong)count * (ulong)Unsafe.SizeOf<T>();
                            System.Buffer.MemoryCopy(srcPtr, dst + index * (ulong)Unsafe.SizeOf<T>(), 
                                                    bytesToCopy, bytesToCopy);
                        }
                        
                        index += (ulong)count;
                        count = 0;
                        
                        if (index >= data.Length)
                        {
                            break;
                        }
                    }
                }
            }
            finally
            {
                // 3. Unmap memory
                vk.UnmapMemory(device, memory);
            }
        }
    }

    public unsafe void UploadDeviceConstants<T>(ref Buffer buffer, T[] data, ulong start, ulong count) where T : struct
    {
        ulong typeSize = (ulong)Unsafe.SizeOf<T>();
        ulong dataSize = count * typeSize;
        var dstOffset = start * typeSize;

        // 1. Create staging buffer (host-visible)
        CreateBuffer<T>(
            count,
            BufferUsageFlags.TransferSrcBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out var stagingBuffer,
            out var stagingMemory);

        // 2. Upload to staging memory manually
        void* mapped = null;
        var result = vk.MapMemory(device, stagingMemory, 0, dataSize, 0, &mapped);
        if (result != Result.Success)
            throw new Exception("Failed to map staging buffer for UploadConstants.");

        try
        {
            byte* dst = (byte*)mapped;
            for (var i = 0ul; i < count; i++)
            {
                Unsafe.Write(dst + i * typeSize, data[start + i]);
            }
        }
        finally
        {
            vk.UnmapMemory(device, stagingMemory);
        }

        // 3. Copy to device-local buffer
        var cmd = BeginSingleTimeCommands(commandPool);
        BufferCopy copyRegion = new()
        {
            SrcOffset = 0,
            DstOffset = dstOffset,
            Size = dataSize
        };

        vk.CmdCopyBuffer(cmd, stagingBuffer, buffer, 1, &copyRegion);
        EndSingleTimeCommands(cmd, this.computeQueue, commandPool);

        // 4. Cleanup
        vk.DestroyBuffer(device, stagingBuffer, null);
        vk.FreeMemory(device, stagingMemory, null);
    }

    public unsafe void UploadConstants<T>(ref DeviceMemory memory, T[] data, int start, int count) where T : struct
    {
        ulong dataSize = (ulong)count * (ulong)Unsafe.SizeOf<T>();
        void* mappedMemory = null;

        // 1. Map memory
        var res = vk.MapMemory(device, memory, 0, dataSize, 0, &mappedMemory);
        if (res != Result.Success)
        {
            throw new Exception("Failed to map Vulkan memory.");
        }

        mappedMemory = (byte*)mappedMemory + (ulong)start * (ulong)Unsafe.SizeOf<T>();
        // 2. Copy data manually
        foreach (var item in data)
        {
            Unsafe.Write(mappedMemory, item);
            mappedMemory = (byte*)mappedMemory + Unsafe.SizeOf<T>();
        }

        // 3. Unmap memory
        vk.UnmapMemory(device, memory);
    }

    public unsafe void UploadDeviceConstants<T>(ref Buffer buffer, T[] data) where T : struct
    {
        UploadDeviceConstants(ref buffer, data, 0, (ulong)data.Length);
    }

    /// <summary>
    /// Uploads data to a Vulkan buffer (host-visible).
    /// </summary>
    public unsafe void UploadConstants<T>(ref DeviceMemory memory, T[] data) where T : struct
    {
        UploadConstants(ref memory, data, 0, data.Length);
    }

    public unsafe T[] ReadDeviceBufferData<T>(ref Buffer buffer, ulong elementCount) where T : struct
    {
        //Create staging buffer
        CreateBuffer<T>(
            elementCount,
            BufferUsageFlags.TransferDstBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out var stagingBuffer,
            out var stagingMemory);
        
        CopyBuffer(ref buffer, ref stagingBuffer, elementCount * (ulong)Unsafe.SizeOf<T>());

        // 3. Read from the staging buffer
        T[] result = ReadBufferData<T>(ref stagingMemory, elementCount); // reuse your method here

        // 4. Cleanup
        vk.DestroyBuffer(device, stagingBuffer, null);
        vk.FreeMemory(device, stagingMemory, null);

        return result;
    }

    public unsafe void CopyBuffer(ref Buffer srcBuffer, ref Buffer dstBuffer, ulong size)
    {
        var cmd = BeginSingleTimeCommands(this.commandPool);

        BufferCopy copyRegion = new()
        {
            SrcOffset = 0,
            DstOffset = 0,
            Size = size
        };

        vk.CmdCopyBuffer(cmd, srcBuffer, dstBuffer, 1, &copyRegion);
        EndSingleTimeCommands(cmd, this.computeQueue, commandPool);
    }

    public unsafe void EndSingleTimeCommands(
    CommandBuffer commandBuffer,
    Queue queue,
    CommandPool commandPool)
    {
        vk.EndCommandBuffer(commandBuffer);

        SubmitInfo submitInfo = new()
        {
            SType = StructureType.SubmitInfo,
            CommandBufferCount = 1,
            PCommandBuffers = &commandBuffer
        };

        vk.QueueSubmit(queue, 1, &submitInfo, default);
        vk.QueueWaitIdle(queue); // <-- Wait for the copy to finish

        vk.FreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    public unsafe CommandBuffer BeginSingleTimeCommands(CommandPool commandPool)
    {
        CommandBufferAllocateInfo allocInfo = new()
        {
            SType = StructureType.CommandBufferAllocateInfo,
            Level = CommandBufferLevel.Primary,
            CommandPool = commandPool,
            CommandBufferCount = 1
        };

        CommandBuffer commandBuffer;
        vk.AllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        CommandBufferBeginInfo beginInfo = new()
        {
            SType = StructureType.CommandBufferBeginInfo,
            Flags = CommandBufferUsageFlags.OneTimeSubmitBit
        };

        vk.BeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    /// <summary>
    /// Reads data from a Vulkan buffer back to CPU.
    /// </summary>
    public unsafe T[] ReadBufferData<T>(ref DeviceMemory memory, ulong elementCount) where T : struct
    {
        ulong dataSize = elementCount * (ulong)Unsafe.SizeOf<T>();
        void* mappedMemory = null;
        ulong typeSize = (ulong)Unsafe.SizeOf<T>();

        // 1. Map memory
        if (vk.MapMemory(device, memory, 0, dataSize, 0, &mappedMemory) != Result.Success)
        {
            throw new Exception("Failed to map Vulkan memory.");
        }

        // 2. Read data manually
        T[] result = new T[elementCount];
        /*fixed (T* resultPtr = result)
        {
            byte* src = (byte*)mappedMemory;
            byte* dst = (byte*)resultPtr;

            for (ulong i = 0; i < dataSize; i++)
            {
                dst[i] = src[i];
            }
        }*/

        for (var i = 0ul; i < elementCount; i++)
        {
            unsafe
            {
                var ptr = (T*)((byte*)mappedMemory + i * typeSize);
                result[i] = *ptr;
            }
        }

        // 3. Unmap memory
        vk.UnmapMemory(device, memory);
        return result;
    }

    /// <summary>
    /// Finds a memory type that fits required properties.
    /// </summary>
    private unsafe uint FindMemoryType(uint typeFilter, MemoryPropertyFlags properties)
    {
        PhysicalDeviceMemoryProperties memProperties;
        vk.GetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (int i = 0; i < memProperties.MemoryTypeCount; i++)
        {
            if (((typeFilter >> i) & 1) == 1)
            {
                if ((memProperties.MemoryTypes[i].PropertyFlags & properties) == properties)
                {
                    return (uint)i;
                }
            }
        }

        throw new Exception("Failed to find suitable memory type!");
    }

    /// <summary>
    /// Finds the queue family that supports compute operations.
    /// </summary>
    private unsafe uint FindComputeQueueFamily(PhysicalDevice physDevice)
    {
        uint queueFamilyCount = 0;
        vk.GetPhysicalDeviceQueueFamilyProperties(physDevice, &queueFamilyCount, null);

        QueueFamilyProperties[] queueFamilies = new QueueFamilyProperties[queueFamilyCount];
        fixed (QueueFamilyProperties* queueFamiliesPtr = queueFamilies)
        {
            vk.GetPhysicalDeviceQueueFamilyProperties(physDevice, &queueFamilyCount, queueFamiliesPtr);
        }

        for (uint i = 0; i < queueFamilyCount; i++)
        {
            if ((queueFamilies[i].QueueFlags & QueueFlags.ComputeBit) != 0)
            {
                return i;  // Found compute queue
            }
        }

        throw new Exception("Failed to find a compute-capable queue family!");
    }

    /// <summary>
    /// Creates a command pool for the compute queue family.
    /// </summary>
    private unsafe void CreateCommandPool()
    {
        CommandPoolCreateInfo poolInfo = new()
        {
            SType = StructureType.CommandPoolCreateInfo,
            QueueFamilyIndex = computeQueueFamilyIndex,
            Flags = CommandPoolCreateFlags.ResetCommandBufferBit
        };

        if (vk.CreateCommandPool(device, &poolInfo, null, out commandPool) != Result.Success)
        {
            throw new Exception("Failed to create command pool!");
        }
    }

    /// <summary>
    /// Submits a command buffer to the compute queue and waits for execution to complete.
    /// </summary>
    public unsafe void SubmitToComputeQueue(ref CommandBuffer commandBuffer)
    {
        SubmitInfo submitInfo = new()
        {
            SType = StructureType.SubmitInfo,
            CommandBufferCount = 1,
            PCommandBuffers = (CommandBuffer*)Unsafe.AsPointer(ref commandBuffer)
        };

        // Create a fence to wait for execution completion
        FenceCreateInfo fenceInfo = new() { SType = StructureType.FenceCreateInfo };
        Fence fence;
        vk.CreateFence(device, &fenceInfo, null, out fence);

        // Submit command buffer
        vk.QueueSubmit(computeQueue, 1, &submitInfo, fence);
        vk.WaitForFences(device, 1, &fence, true, ulong.MaxValue);

        // Destroy the fence after use
        vk.DestroyFence(device, fence, null);
    }

    /// <summary>
    /// Cleans up all allocated buffers and memory, plus the command pool.
    /// </summary>
    public unsafe void Dispose()
    {
        GC.SuppressFinalize(this);

        // Destroy the command pool
        if (commandPool.Handle != 0)
        {
            vk.DestroyCommandPool(device, commandPool, null);
        }
    }
}
