using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NFAI.Core;

public class ComputeCollection<T>(Stream dataStream, ulong length, long offset) : AbstractComputeCollection(dataStream, length, offset), IDisposable where T : struct
{
    public static ComputeCollection<T> FromEnumerable(IEnumerable<T> data, ulong[]? shape = null, string name = "ComputeCollection")
    {
        if (shape == null)
        {
            shape = new ulong[1];
            shape[0] = (ulong)data.Count();
        }

        var stream = new MemoryStream();
        foreach (var item in data)
        {
            var localItem = item;
            var bytes = MemoryMarshal.AsBytes(MemoryMarshal.CreateSpan(ref localItem, 1));
            stream.Write(bytes);
        }
        stream.Seek(0, SeekOrigin.Begin);
        return new ComputeCollection<T>(stream, (ulong)data.Count(), 0) { Name = name, Shape = shape, ConstantCount = 0 };
    }

    public override ushort TypeSize { get; set; } = (ushort)Unsafe.SizeOf<T>();
    //public new List<float> Data => base.Data.Select(x => BitConverter.ToSingle(x, 0)).ToList();

    public IEnumerable<T> GetData()
    {
        /*await foreach (var data in base.GetData())
        {
            yield return MemoryMarshal.Cast<byte, T>(data.Span)[0];
        }*/
        return base.GetData<T>();
    }

    public new IEnumerable<byte> GetDataRaw()
    {
        return base.GetDataRaw();
    }

    public void Dispose()
    {
        GC.SuppressFinalize(this);
        //base.DataStream.Dispose();
    }

    public static ComputeCollection<T> Cast<TFrom>(ComputeCollection<TFrom> collection) where TFrom : struct
    {
        var cc = new ComputeCollection<T>(collection.DataStream, collection.Length, collection.offset)
        {
            Shape = collection.Shape,
            ConstantCount = collection.ConstantCount,
            Name = collection.Name,
            TypeSize = collection.TypeSize
        };
        return cc;
    }
}
