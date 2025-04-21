using System.Diagnostics;
using System.Runtime.InteropServices;

namespace NFAI.Internal;

public abstract class AbstractComputeCollection
{
    protected Stream DataStream { get; private init; }
    public virtual ushort TypeSize { get; set; }
    public required virtual ulong[] Shape { get; init; } = [];
    public ulong Length { get; private init; }
    public required ulong ConstantCount { get; init; }
    public long offset { get; private init; }
    public required string Name { get; init; }

    public AbstractComputeCollection(Stream dataStream, ulong length, long offset)
    {
        Length = length;
        //dataStream.CopyTo(newStream, (int)Length + (int)ConstantCount * 4);
        this.offset = offset;
        DataStream = dataStream;
    }

    public async IAsyncEnumerable<float> GetConstants()
    {
        DataStream.Seek(offset + (long)Length * (long)TypeSize, SeekOrigin.Begin);
        //DataStream.Seek(offset, SeekOrigin.Begin);
        var buffer = new byte[4];
        for (var i = 0ul; i < ConstantCount; i++)
        {
            await DataStream.ReadExactlyAsync(buffer.AsMemory(0, 4));
            yield return BitConverter.ToSingle(buffer, 0);
        }
    }

    protected IEnumerable<byte> GetDataRaw()
    {
        DataStream.Seek(offset, SeekOrigin.Begin);
        const int batchSizeInBytes = 1024 * 1024 * 1024; // 4KB batch size, adjust as needed
        int itemsPerBatch = batchSizeInBytes;
        if (itemsPerBatch < 1) itemsPerBatch = 1;
        
        var current = 0ul;
        var total = Length * TypeSize;
        var batchBuffer = new byte[itemsPerBatch];
        // Read the entire batch
        //Memory<byte> batchMemory  = new Memory<byte>(batchBuffer);
        while (current < total)
        {
            if (total > 1000000 && current != 0)
            {
                Console.WriteLine($"Progress: {current / (float)total}");
            }
                
            // Calculate remaining items and adjust batch size if needed
            var itemsToRead = Math.Min(itemsPerBatch, (int)(total - current));
            //var bytesToRead = itemsToRead * TypeSize;
            
            try
            {
                DataStream.ReadExactly(batchBuffer, 0, itemsToRead);
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }
            current += (ulong)itemsToRead;
            foreach (var b in batchBuffer)
            {
                yield return b;
            }
        }
    }

    protected IEnumerable<T> GetData<T>() where T : struct
    {
        DataStream.Seek(offset, SeekOrigin.Begin);
        const int batchSizeInBytes = 1024 * 1024 * 100; // 4KB batch size, adjust as needed
        int itemsPerBatch = batchSizeInBytes / TypeSize;
        if (itemsPerBatch < 1) itemsPerBatch = 1;
        
        var current = 0ul;
        var total = Length;
        var batchBuffer = new byte[itemsPerBatch * TypeSize];
        // Read the entire batch
        //Memory<byte> batchMemory  = new Memory<byte>(batchBuffer);
        while (current < total)
        {
            if (total > 1000000 && current != 0)
            {
                Console.WriteLine($"Progress: {current / (float)total}");
            }
                
            // Calculate remaining items and adjust batch size if needed
            var itemsToRead = Math.Min(itemsPerBatch, (int)(total - current));
            //var bytesToRead = itemsToRead * TypeSize;
            
            try
            {
                DataStream.ReadExactly(batchBuffer, 0, itemsToRead * TypeSize);
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }

            var sw = Stopwatch.StartNew();
            T[] outputs;

            if (TypeSize == 2 && typeof(T) == typeof(float))
            {
                var shorts = MemoryMarshal.Cast<byte, Half>(batchBuffer).ToArray();
                outputs = new T[itemsToRead];
                for (var j = 0; j < itemsToRead; j++)
                {
                    outputs[j] = (T)(object)(float)shorts[j];
                }
            }
            else
            {
                outputs = MemoryMarshal.Cast<byte, T>(batchBuffer).ToArray();
            }

            for (var j = 0; j < itemsToRead; j++)
            {
                yield return outputs[j];
            }

            sw.Stop();
            Console.WriteLine($"Batch time: {sw.ElapsedMilliseconds}ms");
            current += (ulong)itemsToRead;
        }
    }
}
