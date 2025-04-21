namespace NFAI.Internal;

public class TensorInfo
{
    public required string Name { get; init; }
    public required List<UInt64> Shape { get; init; }
    public required string DataType { get; init; }
    public required UInt64 Offset { get; init; }
    public required int Constants { get; init; }
}