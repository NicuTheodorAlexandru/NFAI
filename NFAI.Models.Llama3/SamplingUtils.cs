namespace NFAI.Models.Llama3;

public static class SamplingUtils
{
    public static uint TopP(float[] values, float temperature = 0.6f, float topP = 0.9f, int topK = 40)
    {
        List<float> scaledLogits = values.Select(l => l / temperature).ToList();
        List<float> probs = Softmax(scaledLogits);
        var indexedProbs = probs
            .Select((p, i) => new { Index = i, Prob = p })
            .OrderByDescending(x => x.Prob)
            .ToList();
        var finalIndexedProbs = indexedProbs.Take(topK).ToList();
        float cumulative = 0f;
        List<(int idx, float prob)> topPList = [];
        foreach (var x in finalIndexedProbs)
        {
            cumulative += x.Prob;
            topPList.Add((x.Index, x.Prob));
            if (cumulative >= topP) break;
        }
        float total = topPList.Sum(x => x.prob);
        var normalized = topPList.Select(x => (x.idx, prob: x.prob / total)).ToList();
        float rand = Random.Shared.NextSingle();
        float running = 0f;
        foreach (var (idx, prob) in normalized)
        {
            running += prob;
            if (rand < running)
                return (uint)idx;
        }
        return (uint)normalized[^1].idx;
    }

    public static List<float> Softmax(List<float> logits)
    {
        float maxLogit = logits.Max();
        var exps = logits.Select(l => MathF.Exp(l - maxLogit)).ToList();
        float sumExp = exps.Sum();
        return exps.Select(e => e / sumExp).ToList();
    }

    public static uint ArgMax(float[] values)
    {
        var topPairs = values
            .Select((value, index) => new { Value = value, Index = index })
            .OrderByDescending(x => x.Value)
            .Take(10)
            .ToList();
        Console.WriteLine("Top 10 predictions:");
        foreach (var pair in topPairs)
        {
            Console.WriteLine($"Index {pair.Index} → Value {pair.Value}");
        }
        var max = values.Max();
        return (uint)values.ToList().IndexOf(max);
    }
}
