namespace NFAI.Models;

public class ModelOptions
{
    public string GGUFPath { get; set; } = string.Empty;

    public uint KVCacheSize { get; set; } = 512;
}
