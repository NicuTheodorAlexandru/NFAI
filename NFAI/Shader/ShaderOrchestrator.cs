namespace NFAI.Shader;

public class ShaderOrchestrator<TStartShader, TEndShader>
    where TStartShader : ShaderWrapper
    where TEndShader : ShaderWrapper
{
    private readonly List<ShaderWrapper> shaders = [];

    public ShaderOrchestrator(TStartShader startShader, TEndShader endShader, params ShaderWrapper[] middleShaders)
    {
        shaders.Add(startShader);
        shaders.AddRange(middleShaders);
        shaders.Add(endShader);
    }
}