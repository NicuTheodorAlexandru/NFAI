using NFAI.Internal;
using Silk.NET.Vulkan;

namespace NFAI.Shader;

public class SplitterShader<T> : ShaderWrapper where T : struct
{
    // Constants for work group sizing
    private const uint LOCAL_SIZE_X = 64; // Work group size for sequence dimension
    private const uint LOCAL_SIZE_Y = 1;  // Work group size for head dimension

    private readonly ShaderProperty<T> inputData;
    private readonly ShaderProperty<T> outputData;
    private readonly ShaderProperty<uint> inputSize;
    private readonly ShaderProperty<uint> headCount;
    private readonly ShaderProperty<uint> headDim;

    public SplitterShader(
        Vk vk, 
        Device device, 
        VulkanBufferManager vulkanBufferManager, 
        uint inputSize,
        uint headCount,
        uint headDim) 
        : base(vk, device, vulkanBufferManager, "Splitter")
    {
        // Create shader properties
        this.inputData = new ShaderProperty<T>(vulkanBufferManager, VulkanTransferType.StorageBuffer, inputSize)
        {
            Name = "inputData",
            IsCollection = true
        };
        
        // Output size is the same as input size, we're just reshaping/reorganizing the data
        this.outputData = new ShaderProperty<T>(vulkanBufferManager, VulkanTransferType.StorageBuffer, inputSize)
        {
            Name = "outputData",
            IsCollection = true
        };
        
        this.inputSize = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "inputSize",
            IsCollection = false
        };
        
        this.headCount = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "headCount",
            IsCollection = false
        };
        
        this.headDim = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "headDim",
            IsCollection = false
        };

        // Add properties to the shader
        AddProperties(this.inputData, this.outputData, this.inputSize, this.headCount, this.headDim);
        
        // Set default values
        this.inputSize.SetValue([inputSize]);
        this.headCount.SetValue([headCount]);
        this.headDim.SetValue([headDim]);

        // Add extensions if needed
        extensions.Add("GL_EXT_debug_printf");
    }

    public T[] GetOutputs()
    {
        return outputData.GetValue();
    }

    public ShaderProperty<T> GetOutputProperty()
    {
        return outputData;
    }

    public override void Compute(AbstractComputeCollection computeCollection)
    {
        if (computeCollection is ComputeCollection<T> collection)
        {
            Compute(collection);
        }
        else
        {
            throw new ArgumentException($"Expected ComputeCollection<{typeof(T).Name}> but got {computeCollection.GetType().Name}");
        }
    }

    public void Compute(ComputeCollection<T> value)
    {
        inputData.SetValue(value);
        
        // Calculate dispatch dimensions based on sequence length and head dimensions
        uint totalSize = inputSize.GetValue()[0];
        uint heads = headCount.GetValue()[0];
        
        uint groupsX = (totalSize + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
        
        base.Compute(groupsX, 1, 1);
    }

    public void Compute(ShaderProperty<T> value)
    {
        // Connect input shader property to our inputData property
        value.TransferTo(inputData);
        
        // Calculate dispatch dimensions
        uint totalSize = inputSize.GetValue()[0];
        uint heads = headCount.GetValue()[0];
        
        uint groupsX = (totalSize + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
        
        base.Compute(groupsX, 1, 1);
    }

    public (uint x, uint y, uint z) CalculateDispatchSize()
    {
        uint totalSize = inputSize.GetValue()[0];
        uint groupsX = (totalSize + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
        return (groupsX, 1, 1);
    }

    protected override string GetMainMethodCode()
    {
        return @"
        // Define local work group sizes
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        
        void main()
        {
            uint index = gl_GlobalInvocationID.x;
            
            // Early exit if out of bounds
            if (index >= inputsize) {
                return;
            }
            
            // Calculate which head this element belongs to
            uint head = index / headdim;
            
            // Calculate position within the head
            uint posInHead = index % headdim;
            
            // Calculate output position
            // Format: [head0_all_values, head1_all_values, ...] -> [head0_dim0, head1_dim0, ..., head0_dim1, head1_dim1, ...]
            // This organizes data so all heads' same dimension are adjacent
            uint outIndex = posInHead * headcount + head;
            
            // Bounds check
            if (outIndex < inputsize) {
                outputdata[outIndex] = inputdata[index];
            } else {
                // Debug printf if something goes wrong
                // debugPrintfEXT(""Out of bounds: index=%u, head=%u, posInHead=%u, outIndex=%u"", 
                //                index, head, posInHead, outIndex);
            }
        }";
    }
}