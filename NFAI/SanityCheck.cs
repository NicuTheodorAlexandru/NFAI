namespace NFAI;

public static class SanityCheck
{
    private static string DebugFolderPath => @"E:\AiHome\GGUF test\Debug";

    public static void Logits(float[] actual, uint tokenIndex)
    {
        var blockFolderPath = Path.Combine(DebugFolderPath, $"final_logits");
        var path = Path.Combine(blockFolderPath, $"final_logits_{tokenIndex}.txt");

        using var reader = new StreamReader(File.OpenRead(path));
        var attnScores = new List<float>();
        var sep = ',';
        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (line != null)
            {
                var values = line.Split(sep);
                foreach (var value in values)
                {
                    if (float.TryParse(value, out var score))
                    {
                        attnScores.Add(score);
                    }
                }
            }
        }

        if (attnScores.Count != actual.Count())
        {
            Console.Error.WriteLine($"Attention values count mismatch for token {tokenIndex}.");
        }

        for (var i = 0; i < attnScores.Count; i++)
        {
            if (MathF.Abs(attnScores[i] - actual.ElementAt(i)) > 0.0001f)
            {
                Console.Error.WriteLine($"Attention values mismatch at index {i} for token {tokenIndex}: {attnScores[i]} != {actual.ElementAt(i)}");
            }
        }

    }

    public static void AttnScoreCalc(IEnumerable<float> actual, uint blockIndex, uint tokenIndex, string name)
    {
        var blockFolderPath = Path.Combine(DebugFolderPath, $"block_{blockIndex}");
        var folderPath = Path.Combine(blockFolderPath, $"attention");
        var f1Path = Path.Combine(folderPath, $"{name}");
        var path = Path.Combine(f1Path, $"{name}_{tokenIndex}.txt");

        using var reader = new StreamReader(File.OpenRead(path));
        var attnScores = new List<float>();
        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (line != null && float.TryParse(line, out var value))
            {
                attnScores.Add(value);
            }
        }

        if (attnScores.Count != actual.Count())
        {
            Console.Error.WriteLine($"Attention values count mismatch for block {blockIndex}, token {tokenIndex}.");
        }

        for (var i = 0; i < attnScores.Count; i++)
        {
            if (MathF.Abs(attnScores[i] - actual.ElementAt(i)) > 0.0001f)
            {
                Console.Error.WriteLine($"Attention values mismatch at index {i} for block {blockIndex}, token {tokenIndex}: {attnScores[i]} != {actual.ElementAt(i)}");
            }
        }
    }

    public static void AttnTransposed(IEnumerable<float> actual, uint blockIndex, uint tokenIndex, string name)
    {
        var blockFolderPath = Path.Combine(DebugFolderPath, $"block_{blockIndex}");
        var folderPath = Path.Combine(blockFolderPath, $"attention");
        var f1Path = Path.Combine(folderPath, $"{name}");
        var path = Path.Combine(f1Path, $"{name}_{tokenIndex}.txt");

        using var reader = new StreamReader(File.OpenRead(path));
        var attnSdpa = new List<float>();
        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (line != null && float.TryParse(line, out var value))
            {
                attnSdpa.Add(value);
            }
        }

        if (attnSdpa.Count != actual.Count())
        {
            Console.Error.WriteLine($"Attention values count mismatch for block {blockIndex}, token {tokenIndex}.");
        }

        for (var i = 0; i < actual.Count(); i++)
        {
            if (MathF.Abs(attnSdpa[i] - actual.ElementAt(i)) > 0.0001f)
            {
                Console.Error.WriteLine($"Attention values mismatch at index {i} for block {blockIndex}, token {tokenIndex}: {attnSdpa[i]} != {actual.ElementAt(i)}");
            }
        }
    }

    public static void AttnSdpa(IEnumerable<float> actual, uint blockIndex, uint tokenIndex)
    {
        var blockFolderPath = Path.Combine(DebugFolderPath, $"block_{blockIndex}");
        var folderPath = Path.Combine(blockFolderPath, $"attention");
        var f1Path = Path.Combine(folderPath, $"attn_sdpa_reshape");
        var path = Path.Combine(f1Path, $"attn_sdpa_reshape_{tokenIndex}.txt");

        using var reader = new StreamReader(File.OpenRead(path));
        var attnSdpa = new List<float>();
        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (line != null && float.TryParse(line, out var value))
            {
                attnSdpa.Add(value);
            }
        }

        if (attnSdpa.Count != actual.Count())
        {
            Console.Error.WriteLine($"Attention values count mismatch for block {blockIndex}, token {tokenIndex}.");
        }

        for (var i = 0; i < attnSdpa.Count; i++)
        {
            if (MathF.Abs(attnSdpa[i] - actual.ElementAt(i)) > 0.0001f)
            {
                Console.Error.WriteLine($"Attention values mismatch at index {i} for block {blockIndex}, token {tokenIndex}: {attnSdpa[i]} != {actual.ElementAt(i)}");
                var toFind = attnSdpa[i];
                // Find all occurrences of similar values in the *entire* actual list and their indices
                var similarOccurrences = actual
                    .Skip(i) // Skip to the current index
                    .Select((value, index) => new { Value = value, Index = index }) // Get value and index
                    .Where(item => MathF.Abs(item.Value - toFind) < 0.0001f)       // Filter by similarity
                    .ToList();                                                     // Materialize the results

                if (similarOccurrences.Count > 0)
                {
                    // Format the output to show value and index for each occurrence
                    var formattedSimilar = similarOccurrences.Select(item => $"{item.Value} (at index {item.Index + i})");
                    Console.Error.WriteLine($"Expected value {toFind} found elsewhere in actual results at: {string.Join(", ", formattedSimilar)}");
                }
            }
        }
    }

    public static void AttnRopeQueries(IEnumerable<float> actual, uint blockIndex, uint tokenIndex, float[] freqs, float[] initial)
    {
        var blockFolderPath = Path.Combine(DebugFolderPath, $"block_{blockIndex}");
        var folderPath = Path.Combine(blockFolderPath, $"attention");
        var f1Path = Path.Combine(folderPath, $"attn_xq_rope");
        var path = Path.Combine(f1Path, $"attn_xq_rope_{tokenIndex}.txt");

        using var reader = new StreamReader(File.OpenRead(path));
        var attnRopeQueries = new List<float>();
        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (line != null && float.TryParse(line, out var value))
            {
                attnRopeQueries.Add(value);
            }
        }

        if (attnRopeQueries.Count != actual.Count())
        {
            Console.Error.WriteLine($"Attention values count mismatch for block {blockIndex}, token {tokenIndex}.");
        }

        for (var i = 0; i < attnRopeQueries.Count; i++)
        {
            if (MathF.Abs(attnRopeQueries[i] - actual.ElementAt(i)) > 0.0001f)
            {
                Console.Error.WriteLine($"Attention values mismatch at index {i} for block {blockIndex}, token {tokenIndex}: {attnRopeQueries[i]} != {actual.ElementAt(i)}");
                var a = initial[i];
                var b = initial[i + 1];
                // find right theta
                for (var j = 0; j < freqs.Length; j++)
                {
                    var theta = freqs[j] * tokenIndex;
                    var (value1, value2) = RoPE(a, b, theta);
                    if (MathF.Abs(value1 - attnRopeQueries[i]) < 0.0001f && MathF.Abs(value2 - attnRopeQueries[i + 1]) < 0.0001f)
                    {
                        Console.Error.WriteLine($"Found theta: {theta} for index {i}");
                        break;
                    }
                }
                // find right pair
                var t = (i / 2) % freqs.Length;
                for (var pairIndex = 0; pairIndex < attnRopeQueries.Count / 2; pairIndex++)
                {
                    var (a1, b1) = RoPE(a, b, freqs[t] * tokenIndex);
                    var a2 = attnRopeQueries[i];
                    var b2 = attnRopeQueries[i + 1];
                    if (MathF.Abs(a1 - a2) < 0.0001f && MathF.Abs(b1 - b2) < 0.0001f)
                    {
                        Console.Error.WriteLine($"Found pair: {pairIndex} for index {i}");
                        break;
                    }
                }
            }
        }
    }

    private static (float, float) RoPE(float a, float b, float theta)
    {
        theta *= 180f / MathF.PI;
        var cosTheta = MathF.Cos(theta);
        var sinTheta = MathF.Sin(theta);
        var value1 = cosTheta * a - sinTheta * b;
        var value2 = sinTheta * a + cosTheta * b;
        return (value1, value2);
    }

    public static void AttnRopeKeys(IEnumerable<float> actual, uint blockIndex, uint tokenIndex)
    {
        var blockFolderPath = Path.Combine(DebugFolderPath, $"block_{blockIndex}");
        var folderPath = Path.Combine(blockFolderPath, $"attention");
        var f1Path = Path.Combine(folderPath, $"attn_xk_rope");
        var path = Path.Combine(f1Path, $"attn_xk_rope_{tokenIndex}.txt");

        using var reader = new StreamReader(File.OpenRead(path));
        var attnRopeKeys = new List<float>();
        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (line != null && float.TryParse(line, out var value))
            {
                attnRopeKeys.Add(value);
            }
        }

        if (attnRopeKeys.Count != actual.Count())
        {
            Console.Error.WriteLine($"Attention values count mismatch for block {blockIndex}, token {tokenIndex}.");
        }

        for (var i = 0; i < attnRopeKeys.Count; i++)
        {
            if (MathF.Abs(attnRopeKeys[i] - actual.ElementAt(i)) > 0.0001f)
            {
                Console.Error.WriteLine($"Attention values mismatch at index {i} for block {blockIndex}, token {tokenIndex}: {attnRopeKeys[i]} != {actual.ElementAt(i)}");
            }
        }
    }

    public static void AttnValues(IEnumerable<float> actual, uint blockIndex, uint tokenIndex)
    {
        var blockFolderPath = Path.Combine(DebugFolderPath, $"block_{blockIndex}");
        var folderPath = Path.Combine(blockFolderPath, $"attention");
        var f1Path = Path.Combine(folderPath, $"attn_xv_proj");
        var path = Path.Combine(f1Path, $"attn_xv_proj_{tokenIndex}.txt");

        using var reader = new StreamReader(File.OpenRead(path));
        var attnValues = new List<float>();
        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (line != null && float.TryParse(line, out var value))
            {
                attnValues.Add(value);
            }
        }

        if (attnValues.Count != actual.Count())
        {
            Console.Error.WriteLine($"Attention values count mismatch for block {blockIndex}, token {tokenIndex}.");
        }

        for (var i = 0; i < attnValues.Count; i++)
        {
            if (MathF.Abs(attnValues[i] - actual.ElementAt(i)) > 0.0001f)
            {
                Console.Error.WriteLine($"Attention values mismatch at index {i} for block {blockIndex}, token {tokenIndex}: {attnValues[i]} != {actual.ElementAt(i)}");
            }
        }
    }

    public static void AttnKeys(IEnumerable<float> actual, uint blockIndex, uint tokenIndex)
    {
        var blockFolderPath = Path.Combine(DebugFolderPath, $"block_{blockIndex}");
        var folderPath = Path.Combine(blockFolderPath, $"attention");
        var f1Path = Path.Combine(folderPath, $"attn_xk_proj");
        var path = Path.Combine(f1Path, $"attn_xk_proj_{tokenIndex}.txt");

        using var reader = new StreamReader(File.OpenRead(path));
        var attnKeys = new List<float>();
        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (line != null && float.TryParse(line, out var key))
            {
                attnKeys.Add(key);
            }
        }

        if (attnKeys.Count != actual.Count())
        {
            Console.Error.WriteLine($"Attention keys count mismatch for block {blockIndex}, token {tokenIndex}.");
        }

        for (var i = 0; i < attnKeys.Count; i++)
        {
            if (MathF.Abs(attnKeys[i] - actual.ElementAt(i)) > 0.0001f)
            {
                Console.Error.WriteLine($"Attention keys mismatch at index {i} for block {blockIndex}, token {tokenIndex}: {attnKeys[i]} != {actual.ElementAt(i)}");
            }
        }
    }

    public static void AttnQuery(IEnumerable<float> actual, uint blockIndex, uint tokenIndex)
    {
        var blockFolderPath = Path.Combine(DebugFolderPath, $"block_{blockIndex}");
        var folderPath = Path.Combine(blockFolderPath, $"attention");
        var f1Path = Path.Combine(folderPath, $"attn_xq_proj");
        var path = Path.Combine(f1Path, $"attn_xq_proj_{tokenIndex}.txt");

        using var reader = new StreamReader(File.OpenRead(path));
        var attnQuery = new List<float>();
        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (line != null && float.TryParse(line, out var query))
            {
                attnQuery.Add(query);
            }
        }

        if (attnQuery.Count != actual.Count())
        {
            Console.Error.WriteLine($"Attention query count mismatch for block {blockIndex}, token {tokenIndex}.");
        }

        for (var i = 0; i < attnQuery.Count; i++)
        {
            if (Math.Abs(attnQuery[i] - actual.ElementAt(i)) > 0.0001f)
            {
                Console.Error.WriteLine($"Attention query mismatch at index {i} for block {blockIndex}, token {tokenIndex}: {attnQuery[i]} != {actual.ElementAt(i)}");
            }
        }
    }

    public static void FfnNorm(IEnumerable<float> actual, uint blockIndex, uint tokenIndex)
    {
        var blockFolderPath = Path.Combine(DebugFolderPath, $"block_{blockIndex}");
        var folderPath = Path.Combine(blockFolderPath, $"ffnNorm");
        var path = Path.Combine(folderPath, $"ffnNorm_{tokenIndex}.txt");

        using var reader = new StreamReader(File.OpenRead(path));
        var ffnNorm = new List<float>();
        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (line != null && float.TryParse(line, out var norm))
            {
                ffnNorm.Add(norm);
            }
        }

        if (ffnNorm.Count != actual.Count())
        {
            Console.Error.WriteLine($"FFN norm count mismatch for block {blockIndex}, token {tokenIndex}.");
        }

        for (var i = 0; i < ffnNorm.Count; i++)
        {
            if (MathF.Abs(ffnNorm[i] - actual.ElementAt(i)) > 0.0001f)
            {
                Console.Error.WriteLine($"FFN norm mismatch at index {i} for block {blockIndex}, token {tokenIndex}: {ffnNorm[i]} != {actual.ElementAt(i)}");
            }
        }
    }

    public static void AttnNorm(IEnumerable<float> actual, uint blockIndex, uint tokenIndex)
    {
        var blockFolderPath = Path.Combine(DebugFolderPath, $"block_{blockIndex}");
        var folderPath = Path.Combine(blockFolderPath, $"attnNorm");
        var path = Path.Combine(folderPath, $"attnNorm_{tokenIndex}.txt");

        using var reader = new StreamReader(File.OpenRead(path));
        var attnNorm = new List<float>();
        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (line != null && float.TryParse(line, out var norm))
            {
                attnNorm.Add(norm);
            }
        }

        if (attnNorm.Count != actual.Count())
        {
            Console.Error.WriteLine($"Attention norm count mismatch for block {blockIndex}, token {tokenIndex}.");
        }

        for (var i = 0; i < attnNorm.Count; i++)
        {
            if (MathF.Abs(attnNorm[i] - actual.ElementAt(i)) > 0.0001f)
            {
                Console.Error.WriteLine($"Attention norm mismatch at index {i} for block {blockIndex}, token {tokenIndex}: {attnNorm[i]} != {actual.ElementAt(i)}");
            }
        }
    }

    public static void Embedding(IEnumerable<float> embeddings, uint tokenIndex)
    {
        var folderPath = Path.Combine(DebugFolderPath, $"token_embeddings");
        var path = Path.Combine(folderPath, $"token_embedding_{tokenIndex}.txt");
        using var reader = new StreamReader(File.OpenRead(path));
        var embeds = new List<float>();
        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (line != null && float.TryParse(line, out var embed))
            {
                embeds.Add(embed);
            }
        }

        if (embeds.Count != embeddings.Count())
        {
            Console.Error.WriteLine($"Embedding count mismatch for token {tokenIndex}.");
        }

        for (var i = 0; i < embeds.Count; i++)
        {
            if (embeds[i] != embeddings.ElementAt(i))
            {
                Console.Error.WriteLine($"Embedding mismatch at index {i} for token {tokenIndex}: {embeds[i]} != {embeddings.ElementAt(i)}");
            }
        }
    }

    public static void PromptTokenIds(IEnumerable<uint> tokenIds)
    {
        var path = Path.Combine(DebugFolderPath, "prompt.txt");
        using var reader = new StreamReader(File.OpenRead(path));
        var content = reader.ReadToEnd();
        var stringIds = content.Split(',');

        var tokenIdList = new List<uint>();
        foreach (var stringId in stringIds)
        {
            if (uint.TryParse(stringId, out var tokenId))
            {
                tokenIdList.Add(tokenId);
            }
            else
            {
                Console.Error.WriteLine($"Invalid token ID: {stringId}");
            }
        }

        if (tokenIdList.Count != tokenIds.Count())
        {
            Console.Error.WriteLine("Token ID count mismatch.");
        }

        for (var i = 0; i < tokenIdList.Count; i++)
        {
            if (tokenIdList[i] != tokenIds.ElementAt(i))
            {
                Console.Error.WriteLine($"Token ID mismatch at index {i}: {tokenIdList[i]} != {tokenIds.ElementAt(i)}");
            }
        }
    }
}