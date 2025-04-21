using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using NFAI.Core;
using Silk.NET.Vulkan;

namespace NFAI.Vulkan.Shaders;

public unsafe abstract class ShaderWrapper(Vk vk, Device device, VulkanBufferManager vulkanBufferManager, string shaderFileName)
{
    private DescriptorSetLayoutBinding[] descriptorSets = [];
    private WriteDescriptorSet[] writeDescriptorSets = [];
    private readonly List<BaseShaderProperty> properties = [];
    public ShaderModule ShaderModule { get; private set; }
    private DescriptorSetLayout descriptorSetLayout;
    private DescriptorSet descriptorSet;
    private DescriptorPool descriptorPool;
    private DescriptorPoolSize[]? poolSizes;
    protected List<string> extensions = [];
    private CommandBuffer cmdBuffer;
    private Pipeline pipeline;
    private PipelineLayout pipelineLayout;
    private bool setupDone = false;
    private readonly string randomGuid = Guid.NewGuid().ToString();

    public abstract void Compute(AbstractComputeCollection computeCollection);

    protected void Compute(uint groupX, uint groupY, uint groupZ)
    {
        if (setupDone)
        {
            ComputeInternal(groupX, groupY, groupZ);
            return;
        }
        Compile(groupX, groupY, groupZ);
        // TODO actually run the shader, just run it though, the user should get any data from the ShaderProperty instances

    }

    protected void AddProperties(params BaseShaderProperty[] properties)
    {
        foreach (var property in properties)
        {
            AddProperty(property);
        }
    }

    protected void AddProperty(BaseShaderProperty property)
    {
        properties.Add(property);
    }

    protected abstract string GetMainMethodCode();

    private void Compile(uint groupX, uint groupY, uint groupZ)
    {
        setupDone = true;
        WriteAsCompFile();
        // For example, "Tanh.spv" / "Tanh.comp" in the "Shaders" folder.
        string spvPath = $"Shader\\{shaderFileName}.spv";
        string compPath = $"Shader\\{shaderFileName}.comp";

        if (Directory.Exists("Shader") == false)
        {
            Directory.CreateDirectory("Shader");
        }

        // 1) If the .spv file doesn’t exist, try to compile the .comp file using glslangValidator.
        if (!File.Exists(spvPath))
        {
            Console.WriteLine($"[INFO] {spvPath} not found. Attempting to compile {compPath}...");

            if (!File.Exists(compPath))
            {
                throw new FileNotFoundException($"Shader source .comp file not found: {compPath}");
            }

            // Use an external process to invoke glslangValidator (must be in your PATH)
            var startInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "glslangValidator",
                Arguments = $"-V \"{compPath}\" -o \"{spvPath}\"", // '-V' -> compile to SPIR-V
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = new System.Diagnostics.Process
            {
                StartInfo = startInfo
            };

            process.Start();
            process.WaitForExit();

            if (process.ExitCode != 0)
            {
                string err = process.StandardOutput.ReadToEnd();
                Console.WriteLine($"Failed to compile shader {compPath} -> {spvPath}.\nError:\n{err}");
                throw new Exception($"Failed to compile shader {compPath} -> {spvPath}.");
            }

            Console.WriteLine($"[INFO] Successfully compiled {compPath} -> {spvPath}");
        }

        // 2) Now read the compiled SPIR-V bytes from the .spv file
        byte[] shaderBytes = File.ReadAllBytes(spvPath);

        // 3) Create the Vulkan shader module
        unsafe
        {
            var shaderCode = Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(shaderBytes));
            ShaderModuleCreateInfo createInfo = new()
            {
                SType = StructureType.ShaderModuleCreateInfo,
                CodeSize = (nuint)shaderBytes.Length,
                PCode = (uint*)shaderCode,
            };

            if (vk.CreateShaderModule(device, &createInfo, null, out ShaderModule shaderModule)
                != Result.Success)
            {
                throw new Exception($"Failed to create shader module from {spvPath}");
            }

            ShaderModule = shaderModule;


            DescriptorSetLayoutBinding* bindingsPtr = (DescriptorSetLayoutBinding*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(descriptorSets));

            var layoutInfo = new DescriptorSetLayoutCreateInfo()
            {
                SType = StructureType.DescriptorSetLayoutCreateInfo,
                BindingCount = (uint)descriptorSets.Length,
                PBindings = bindingsPtr
            };

            vk.CreateDescriptorSetLayout(device, &layoutInfo, null, out descriptorSetLayout);

            CreateDescriptorPoolAndSet();

            var pipelineStageInfo = new PipelineShaderStageCreateInfo
            {
                SType = StructureType.PipelineShaderStageCreateInfo,
                Stage = ShaderStageFlags.ComputeBit,
                Module = ShaderModule,
                PName = (byte*)Marshal.StringToHGlobalAnsi("main"),
            };

            PipelineLayoutCreateInfo pipelineLayoutInfo;
            pipelineLayoutInfo = new PipelineLayoutCreateInfo
            {
                SType = StructureType.PipelineLayoutCreateInfo,
                SetLayoutCount = 1,
                PSetLayouts = (DescriptorSetLayout*)Unsafe.AsPointer(ref descriptorSetLayout),
                PushConstantRangeCount = 0,
                PPushConstantRanges = null
            };

            vk.CreatePipelineLayout(device, &pipelineLayoutInfo, null, out pipelineLayout);

            var pipelineInfo = new ComputePipelineCreateInfo
            {
                SType = StructureType.ComputePipelineCreateInfo,
                Stage = pipelineStageInfo,
                Layout = pipelineLayout
            };

            var pipelineCacheInfo = new PipelineCacheCreateInfo
            {
                SType = StructureType.PipelineCacheCreateInfo
            };

            vk.CreatePipelineCache(device, &pipelineCacheInfo, null, out var pipelineCache);

            var res = vk.CreateComputePipelines(device, pipelineCache, 1, &pipelineInfo, null, out pipeline);

            var allocInfo = new CommandBufferAllocateInfo
            {
                SType = StructureType.CommandBufferAllocateInfo,
                CommandPool = vulkanBufferManager.CommandPool,
                Level = CommandBufferLevel.Primary,
                CommandBufferCount = 1
            };

            vk.AllocateCommandBuffers(device, ref allocInfo, out cmdBuffer);

            var writeSetsPtr = (WriteDescriptorSet*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(writeDescriptorSets));

            for (var i = 0; i < writeDescriptorSets.Length; i++)
            {
                var set = writeSetsPtr + i;

                if (set->PBufferInfo == null || set->PBufferInfo->Buffer.Handle == 0)
                {
                    throw new InvalidOperationException($"Buffer handle is null for property {properties[i].GetType().Name}");
                }

                if (set->PBufferInfo->Offset % 16 != 0)
                {
                    throw new InvalidOperationException($"Buffer offset is not a multiple of 16 for property {properties[i].GetType().Name}");
                }
            }

            vk.UpdateDescriptorSets(device, (uint)writeDescriptorSets.Length, writeSetsPtr, 0, null);


            ComputeInternal(groupX, groupY, groupZ);
        }
    }

    private void ComputeInternal(uint groupX, uint groupY, uint groupZ)
    {
        vk.ResetCommandBuffer(cmdBuffer, CommandBufferResetFlags.None);

        var beginInfo = new CommandBufferBeginInfo
        {
            SType = StructureType.CommandBufferBeginInfo,
            Flags = CommandBufferUsageFlags.OneTimeSubmitBit
        };
        vk.BeginCommandBuffer(cmdBuffer, &beginInfo);

        vk.CmdBindPipeline(cmdBuffer, PipelineBindPoint.Compute, pipeline);

        // Ensure all properties have valid buffer handles before updating descriptor sets
        foreach (var property in properties)
        {
            if (property.descriptorBufferInfo->Buffer.Handle == 0)
            {
                throw new InvalidOperationException($"Buffer handle is null for property {property.GetType().Name}");
            }
        }

        vk.CmdBindDescriptorSets(
            cmdBuffer,
            PipelineBindPoint.Compute,
            pipelineLayout,
            0,
            1,
            (DescriptorSet*)Unsafe.AsPointer(ref descriptorSet),
            0,
            null
        );

        vk.CmdDispatch(cmdBuffer, groupX, groupY, groupZ);
        vk.EndCommandBuffer(cmdBuffer);
        
        vulkanBufferManager.SubmitToComputeQueue(ref cmdBuffer);
    }

    private unsafe void WriteAsCompFile()
    {
        // Write the shader code to a .comp file in the "Shaders" folder
        string compPath = $"Shader\\{shaderFileName}.comp";
        var sb = new StringBuilder();
        sb.AppendLine("#version 450");
        foreach (var extension in extensions)
        {
            sb.AppendLine($"#extension {extension} : enable");
        }
        if (!Directory.Exists("Shader"))
        {
            Directory.CreateDirectory("Shader");
        }
        // TODO figure out a way to handle workgroups too
        //sb.AppendLine("layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;");
        var index = 0;
        writeDescriptorSets = new WriteDescriptorSet[properties.Count];
        descriptorSets = new DescriptorSetLayoutBinding[properties.Count];
        for (var i = 0; i < properties.Count; i++)
        {
            if (properties.Count == 0 || properties[i].TransferType == VulkanTransferType.PushConstant)
            {
                continue;
            }

            var descriptorSetLayoutBinding = new DescriptorSetLayoutBinding
            {
                Binding = (uint)i,
                DescriptorType = properties[i].TransferType switch
                {
                    VulkanTransferType.Uniform => DescriptorType.UniformBuffer,
                    VulkanTransferType.StorageBuffer => DescriptorType.StorageBuffer,
                    _ => throw new ArgumentOutOfRangeException()
                },
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.ComputeBit,
            };
            descriptorSets[i] = descriptorSetLayoutBinding;

            WriteDescriptorSet writeDescriptorSet;
            writeDescriptorSet = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet,
                DstBinding = descriptorSetLayoutBinding.Binding,
                DescriptorCount = 1,
                DescriptorType = descriptorSetLayoutBinding.DescriptorType,
                PBufferInfo = properties[i].descriptorBufferInfo,
            };
            writeDescriptorSets[i] = writeDescriptorSet;

            sb.AppendLine(properties[i].GetShaderCode(i));
            index = i;
        }

        var pushConstants = properties.Where(p => p.TransferType == VulkanTransferType.PushConstant).ToArray();
        if (pushConstants.Length > 0)
        {
            sb.AppendLine(@"layout(push_constant) uniform PushConstants");
            sb.AppendLine("{");
            foreach (var prop in pushConstants)
            {
                sb.AppendLine(prop.GetShaderCode(++index));
            }
            sb.AppendLine("} pushConstants;");
            //sb.AppendLine(pushConstants[0].GetShaderCode(index));
        }

        // TODO figure out a way to do the main method better too
        sb.AppendLine(GetMainMethodCode());

        File.WriteAllText(compPath, sb.ToString());
    }

    private unsafe void CreateDescriptorPoolAndSet()
    {
        // We have 5 bindings in a single descriptor set.
        // 4 storage buffers + 1 uniform buffer.

        var uniformCount = properties.Count(p => p.TransferType == VulkanTransferType.Uniform);

        var storageCount = properties.Count(p => p.TransferType == VulkanTransferType.StorageBuffer);

        poolSizes =
        [
            new DescriptorPoolSize { Type = DescriptorType.StorageBuffer, DescriptorCount = (uint)storageCount },
            new DescriptorPoolSize { Type = DescriptorType.UniformBuffer, DescriptorCount = (uint)uniformCount }
        ];

        DescriptorPoolCreateInfo poolInfo = new()
        {
            SType = StructureType.DescriptorPoolCreateInfo,
            PoolSizeCount = (uint)poolSizes.Length,
            PPoolSizes = (DescriptorPoolSize*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(poolSizes)),
            MaxSets = 1
        };

        if (vk.CreateDescriptorPool(device, &poolInfo, null, out descriptorPool) != Result.Success)
        {
            throw new Exception("Failed to create descriptor pool!");
        }

        // Allocate one descriptor set
        DescriptorSetAllocateInfo allocInfo;

        allocInfo = new DescriptorSetAllocateInfo
        {
            SType = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool = descriptorPool,
            DescriptorSetCount = 1,
            PSetLayouts = (DescriptorSetLayout*)Unsafe.AsPointer(ref descriptorSetLayout)
        };

        if (vk.AllocateDescriptorSets(device, ref allocInfo, out descriptorSet) != Result.Success)
        {
            throw new Exception("Failed to allocate descriptor set!");
        }

        for (var i = 0; i < writeDescriptorSets.Length; i++)
        {
            writeDescriptorSets[i].DstSet = descriptorSet;
        }
    }
}