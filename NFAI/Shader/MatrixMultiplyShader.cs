using NFAI.Internal;
using Silk.NET.Vulkan;

namespace NFAI.Shader;

public class MatrixMultiplyShader<TInput, TWeight, TOutput> : ShaderWrapper 
    where TInput : struct
    where TWeight : struct
    where TOutput : struct
{
    // Constants for work group sizing
    private const uint LOCAL_SIZE_X = 4; // Work group size for M dimension
    private const uint LOCAL_SIZE_Y = 64; // Work group size for N dimension

    private readonly ShaderProperty<TInput> inputData;
    private readonly ShaderProperty<TOutput> outputData;
    private readonly ShaderProperty<TWeight> weightData;
    private readonly ShaderProperty<uint> inputRows;
    private readonly ShaderProperty<uint> inputCols;
    private readonly ShaderProperty<uint> outputCols;
    
    // Caching related properties
    private readonly bool useCache;
    private readonly ShaderProperty<uint>? maxContextSize;
    private readonly ShaderProperty<uint>? currentContextSize;
    
    private readonly uint cachedContextSize;
    public uint currentCacheSize;
    private readonly bool transpose;

    public MatrixMultiplyShader(
        Vk vk,
        Device device,
        VulkanBufferManager vulkanBufferManager,
        uint inputRowCount,
        uint inputColCount,
        uint outputColCount,
        ComputeCollection<TWeight>? weights = null,
        uint? contextSize = null,
        bool transpose = true) 
        : base(vk, device, vulkanBufferManager, $"MatrixMultiply_{transpose}_{contextSize}_{inputRowCount}_{inputColCount}_{outputColCount}_{nameof(TInput)}_{nameof(TWeight)}_{nameof(TOutput)}")
    {
        this.transpose = transpose;
        // Create shader properties
        this.inputData = new ShaderProperty<TInput>(vulkanBufferManager, VulkanTransferType.StorageBuffer, inputRowCount * inputColCount)
        {
            Name = "inputData",
            IsCollection = true
        };
        
        // Configure caching if context size is specified
        this.useCache = contextSize.HasValue && contextSize.Value > 0;
        this.cachedContextSize = useCache ? contextSize.Value : 1;
        this.currentCacheSize = 0;
        
        // Output size is [inputRows, outputCols] * contextSize if caching is enabled
        uint outputSize = outputColCount * (useCache ? cachedContextSize : 1);
        this.outputData = new ShaderProperty<TOutput>(vulkanBufferManager, VulkanTransferType.StorageBuffer, outputSize)
        {
            Name = "outputData",
            IsCollection = true
        };
        
        var weightDataSize = 1u;
        if (weights != null)
        {
            weightDataSize = (uint)weights.Length;
        }
        this.weightData = new ShaderProperty<TWeight>(vulkanBufferManager, VulkanTransferType.StorageBuffer, (ulong)weightDataSize)
        {
            Name = "weightData",
            IsCollection = true
        };
        
        this.inputRows = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "inputRows",
            IsCollection = false
        };
        
        this.inputCols = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "inputCols",
            IsCollection = false
        };
        
        this.outputCols = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "outputCols",
            IsCollection = false
        };

        // Add cache-related properties if caching is enabled
        if (useCache)
        {
            this.maxContextSize = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
            {
                Name = "maxContextSize",
                IsCollection = false
            };
            
            this.currentContextSize = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
            {
                Name = "currentContextSize",
                IsCollection = false
            };
                        
            // Add cache-related properties to shader
            AddProperties(this.maxContextSize, this.currentContextSize);

            // Initialize cache-related uniforms    
            this.maxContextSize.SetValue([cachedContextSize]);
            this.currentContextSize.SetValue([0u]);  // Start with empty cache
        }

        // Add main properties to the shader
        AddProperties(this.inputData, this.outputData, this.weightData, 
                     this.inputRows, this.inputCols, this.outputCols);
        
        // Set default values
        this.inputRows.SetValue([inputRowCount]);
        this.inputCols.SetValue([inputColCount]);
        this.outputCols.SetValue([outputColCount]);
        if (weights != null)
        {
            this.weightData.SetValue(weights);
        }
    }

    public TOutput[] GetOutputs()
    {
        return outputData.GetValue();
    }

    public ShaderProperty<TWeight> GetWeightProperty()
    {
        return weightData;
    }

    public ShaderProperty<TInput> GetInputProperty()
    {
        return inputData;
    }

    public ShaderProperty<TOutput> GetOutputProperty()
    {
        return outputData;
    }
    
    public void ResetCache()
    {
        if (!useCache) return;
        
        currentCacheSize = 0;
        currentContextSize?.SetValue([currentCacheSize]);
    }

    public override void Compute(AbstractComputeCollection computeCollection)
    {
        if (computeCollection is ComputeCollection<TInput> compute)
        {
            Compute(compute);
        }
        else if (computeCollection is ShaderProperty<TInput> property)
        {
            Compute(property);
        }
        else
        {
            throw new ArgumentException("Invalid input type");
        }
    }

    public void Compute(ComputeCollection<TInput> value)
    {
        throw new NotImplementedException("Compute with ComputeCollection<TInput> is not implemented. Use Compute(ShaderProperty<TInput>) instead.");
    }

    public void Compute(ComputeCollection<TInput> valueA, ComputeCollection<TWeight> valueB)
    {
        // Connect input shader properties to our inputData property
        inputData.SetValue(valueA);
        weightData.SetValue(valueB);
                
        // Calculate dispatch dimensions based on input and output dimensions
        uint cols = outputCols.GetValue()[0];
        uint inputRows = this.inputRows.GetValue()[0];
        
        uint groupsX = (cols + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
        uint groupsY = (inputRows + LOCAL_SIZE_Y - 1) / LOCAL_SIZE_Y;
                
        base.Compute(groupsX, groupsY, 1);

        // Update cache size tracking for shader
        // Update cache size tracking for shader
        if (useCache)
        {
            // Update current cache size
            currentCacheSize = (currentCacheSize + 1) % cachedContextSize;
            currentContextSize?.SetValue([currentCacheSize]);
        }
    }

    public void Compute(ShaderProperty<TInput> valueA, ShaderProperty<TWeight> valueB)
    {
        // Connect input shader properties to our inputData property
        valueA.TransferTo(inputData);
        valueB.TransferTo(weightData);
        
        // Calculate dispatch dimensions based on input and output dimensions
        uint cols = outputCols.GetValue()[0];
        uint inputRows = this.inputRows.GetValue()[0];
        
        uint groupsX = (cols + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
        uint groupsY = (inputRows + LOCAL_SIZE_Y - 1) / LOCAL_SIZE_Y;
                
        base.Compute(groupsX, groupsY, 1);

        // Update cache size tracking for shader
        if (useCache)
        {
            currentCacheSize++;
            currentContextSize?.SetValue([currentCacheSize]);
        }        
    }

    public void Compute(ShaderProperty<TInput>? value = null)
    {
        // Connect input shader property to our inputData property
        if (value != null)
        {
            value.TransferTo(inputData);
        }
        
        // Calculate dispatch dimensions
        uint cols = outputCols.GetValue()[0];
        uint inputRows = this.inputRows.GetValue()[0];
        
        uint groupsX = (inputRows + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
        uint groupsY = (cols + LOCAL_SIZE_Y - 1) / LOCAL_SIZE_Y;
                
        base.Compute(groupsX, groupsY, 1);

        // Update cache size tracking for shader
        if (useCache)
        {
            currentCacheSize++;
            currentContextSize?.SetValue([currentCacheSize]);
        }        
    }

    protected override string GetMainMethodCode()
    {
        string inputTypeCast = GetTypeCastForGLSL<TInput>();
        string weightTypeCast = GetTypeCastForGLSL<TWeight>();
        string outputTypeCast = GetTypeCastForGLSL<TOutput>();
        string bIndex = !transpose ? $"k * {inputCols.VariableName} + j" : $"j * {inputCols.VariableName} + k";

        return @$"
        // Define local work group sizes
        layout(local_size_x = {LOCAL_SIZE_X}, local_size_y = {LOCAL_SIZE_Y}, local_size_z = 1) in;
        
        void main()
        {{
            // Get global thread indices
            uint i = gl_GlobalInvocationID.x;
            uint j = gl_GlobalInvocationID.y;
            
            // Skip if we're out of bounds
            if (i >= {inputRows.VariableName} || j >= {outputCols.VariableName}) {{
                return;
            }}

            float sum = 0.0;
            for (uint k = 0; k < {inputCols.VariableName}; k++) {{
                // Read input and weight values
                float a = float({inputData.VariableName}[i * {inputCols.VariableName} + k]);
                float b = float({weightData.VariableName}[{bIndex}]);
                sum += a * b;
            }}
            
            // Write result to output
            uint cacheOffset = {(currentContextSize != null ? $"{currentContextSize.VariableName} * {inputRows.VariableName} * {outputCols.VariableName}" : "0")};
            {outputData.VariableName}[i * {outputCols.VariableName} + j + cacheOffset] = {outputTypeCast}(sum);
        }}";
    }

    private string GetTypeCastForGLSL<T>()
    {
        // Add type mappings for GLSL casting based on C# types
        if (typeof(T) == typeof(float))
            return "float";
        else if (typeof(T) == typeof(Half)) // Half precision float
            return "float16_t";
        else if (typeof(T) == typeof(double))
            return "double";
        else if (typeof(T) == typeof(int))
            return "int";
        else if (typeof(T) == typeof(uint))
            return "uint";
        
        // Default to float if unknown type
        return "float";
    }
    
    /// <summary>
    /// Gets the complete cached values calculated so far
    /// </summary>
    /// <returns>An array of all cached values in sequence</returns>
    public TOutput[] GetCurrentCache()
    {
        return outputData.GetValue();
    }
}