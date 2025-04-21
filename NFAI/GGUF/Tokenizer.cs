namespace NFAI.GGUF;

using System.Text;
using System.Text.RegularExpressions;

public class Tokenizer
{
    private readonly List<(byte[], byte[])> merges;
    private readonly Dictionary<byte[], uint> byteSequenceToId = new(new ByteArrayComparer());
    private readonly Dictionary<uint, byte[]> idToByteSequence = [];
    public uint BosTokenId { get; }
    public uint EosTokenId { get; }
    private readonly Regex specialTokenPattern = new(@"<\|.*?\|>");
    private readonly List<char> weirdTokens = [' ', '\n'];

    public Tokenizer(Dictionary<string, object> metadata)
    {
        // Extract tokens from metadata
        var rawTokens = (List<object>)metadata["tokenizer.ggml.tokens"];
        var tokens = rawTokens.Select(x =>
        {
            return x switch
            {
                string s => Encoding.UTF8.GetBytes(s),
                byte[] b => b,
                _ => throw new Exception("Unexpected token format")
            };
        }).ToList();
        
        // Extract merges
        var rawMerges = (List<object>)metadata["tokenizer.ggml.merges"];
        var mergesList = rawMerges.Select(x => {
            var spaceIndex = ((string)x).IndexOf(' ');
            var part1 = ((string)x).Substring(0, spaceIndex);
            var part2 = ((string)x).Substring(spaceIndex + 1);
            return (
                Encoding.UTF8.GetBytes(part1),
                Encoding.UTF8.GetBytes(part2)
            );
        }).ToList();
        
        // Get special token IDs
        this.BosTokenId = (uint)metadata["tokenizer.ggml.bos_token_id"];
        this.EosTokenId = (uint)metadata["tokenizer.ggml.eos_token_id"];
        this.merges = mergesList;

        byteSequenceToId = new Dictionary<byte[], uint>(new ByteArrayComparer());
        idToByteSequence = [];

        for (int i = 0; i < tokens.Count; i++)
        {
            var token = tokens[i];
            var id = (uint)i;

            byteSequenceToId[token] = id;
            idToByteSequence[id] = token;
        }
    }

    private bool IsPotentialHexToken(string token)
    {
        return token.Length == 2 && token.All(c => (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f'));
    }

    /// <summary>
    /// Tokenizes a string into a list of token IDs
    /// </summary>
    public List<uint> Tokenize(string prompt, bool addBos = true, bool addEos = false)
    {
        string template = string.Empty;
        var system_prompt = "You are a helpful assistant.";
        if (addBos)
        {
            template = $@"<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

".Replace("\r", string.Empty);
        }
        else
        {
            template = $@"

<|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

".Replace("\r", string.Empty);
        }

        var specialTokenTemplates = @"<\|[^|>]+?\|>";
        var specialTokenPattern = new Regex(specialTokenTemplates, RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
        var specialMatches = specialTokenPattern.Matches(template);
        // (value1, value2) -> (special token, text until next special token)
        var splitText = new List<(string, string)>();
        var lastIndex = 0;
        for (int i = 0; i < specialMatches.Count; i++)
        {
            var match = specialMatches[i];
            var nextMatchIndex = i + 1 < specialMatches.Count ? specialMatches[i + 1].Index : template.Length;
            string token = match.Value;
            lastIndex = match.Index + token.Length;
            string textUntilNextToken = template.Substring(lastIndex, nextMatchIndex - lastIndex);
            splitText.Add((token, textUntilNextToken));
        }

        var metaTokenPattern = new Regex(
        @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

        var tokenIds = new List<uint>();
        var testMerge = merges.Where(x => x.Item1.Length == 2 && x.Item1[0] == 0xC4 && x.Item1[1] == 0x8A && x.Item2.Length == 2).ToArray();
        foreach(var textPair in splitText)
        {
            var specialTokenBytes = Encoding.UTF8.GetBytes(textPair.Item1);
            var specialTokenId = byteSequenceToId[specialTokenBytes];
            tokenIds.Add(specialTokenId);

            var bytes = Encoding.UTF8.GetBytes(textPair.Item2);

            var matches = metaTokenPattern.Matches(textPair.Item2);
            for (var i = 0; i < matches.Count; i++)
            {
                var match = matches[i];
                var str = match.Value;
                var parts = ToInitialBpeUnits(str);

                while (true)
                {
                    int bestMergeIndex = -1;
                    int bestFirstIndex = -1;
                    List<byte> bestMerged = null;

                    // Scan all adjacent pairs in 'parts'
                    for (int j = 0; j < parts.Count - 1; j++)
                    {
                        var left = parts[j];
                        var right = parts[j + 1];

                        for (int k = 0; k < merges.Count; k++)
                        {
                            var merge = merges[k];
                            if (left.SequenceEqual(merge.Item1) && right.SequenceEqual(merge.Item2))
                            {
                                // Found a valid merge
                                if (bestMergeIndex == -1 || k < bestMergeIndex)
                                {
                                    bestMergeIndex = k;
                                    bestFirstIndex = j;
                                    bestMerged = [.. left, .. right];
                                }
                                break; // stop searching merges once we've found a match for this pair
                            }
                        }
                    }

                    // No more valid merges
                    if (bestMergeIndex == -1)
                        break;

                    // Apply the best merge
                    parts.RemoveAt(bestFirstIndex + 1);
                    parts[bestFirstIndex] = [.. bestMerged];
                }

                foreach (var part in parts)
                {
                    if (byteSequenceToId.TryGetValue(part.ToArray(), out uint id))
                    {
                        tokenIds.Add(id);
                    }
                    else
                    {
                        throw new Exception($"Token not found: {Encoding.UTF8.GetString(part.ToArray())}");
                    }
                }
            }
        }

        return tokenIds;

        // Find special tokens in the text
        /*var specialMatches = FindSpecialTokens(text);

        if (specialMatches.Count > 0)
        {
            int lastEnd = 0;

            foreach (var match in specialMatches)
            {
                int start = match.start;
                int length = match.length;

                // Process the text before the special token
                if (start > lastEnd)
                {
                    string segment = text.Substring(lastEnd, start - lastEnd);
                    byte[] segmentBytes = Encoding.UTF8.GetBytes(segment);
                    tokens.AddRange(TokenizeBytes(segmentBytes));
                }

                // Process the special token itself
                string specialToken = text.Substring(start, length);
                if (tokenToId.TryGetValue(specialToken, out uint id))
                {
                    tokens.Add(id); // Matched whole special token
                }
                else
                {
                    byte[] specialBytes = Encoding.UTF8.GetBytes(specialToken);
                    tokens.AddRange(TokenizeBytes(specialBytes));
                }

                lastEnd = start + length;
            }

            // Process any remaining text after the last special token
            if (lastEnd < text.Length)
            {
                string remaining = text.Substring(lastEnd);
                byte[] finalBytes = Encoding.UTF8.GetBytes(remaining);
                tokens.AddRange(TokenizeBytes(finalBytes));
            }
        }
        else
        {
            // No special tokens, tokenize the whole text
            byte[] textBytes = Encoding.UTF8.GetBytes(text);
            tokens.AddRange(TokenizeBytes(textBytes));
        }

        if (addEos)
        {
            tokens.Add(EosTokenId);
        }

        return tokens;*/
    }

    List<List<byte>> ToInitialBpeUnits(string input)    
    {
        var result = new List<List<byte>>();

        foreach (var ch in input.ToCharArray())
        {
            if (weirdTokens.Contains(ch))
            {
                var bytes = Encoding.UTF8.GetBytes(ch.ToString()).ToList();
                // these tokens seem to be utf8 bytes + an offset of 0x100
                var offset = 0x8A - 0x0A;
                bytes[bytes.Count - 1] += (byte)offset;
                var finalElem = new List<byte>
                {
                    0xC4
                };
                finalElem.AddRange(bytes);
                result.Add(finalElem);
                continue;
            }

            result.Add(Encoding.UTF8.GetBytes(ch.ToString()).ToList());
        }

        return result;
    }

    /// <summary>
    /// Tokenize a sequence of bytes using BPE
    /// </summary>
    /*
    private List<uint> TokenizeBytes(byte[] bytes)
    {
        // First, try to find the exact byte sequence in the vocabulary
        if (byteSequenceToId.TryGetValue(bytes, out uint exactId))
        {
            return [exactId];
        }

        // Start with individual bytes - but preserve multi-byte UTF-8 sequences
        var parts = new List<byte[]>();
        for (int i = 0; i < bytes.Length; i++)
        {
            // Check if this is the start of a multi-byte UTF-8 sequence
            if ((bytes[i] & 0xC0) == 0xC0 && i + 1 < bytes.Length)
            {
                // Determine the length of the UTF-8 sequence
                int sequenceLength = 0;
                if ((bytes[i] & 0xE0) == 0xC0) sequenceLength = 1;      // 2-byte sequence
                else if ((bytes[i] & 0xF0) == 0xE0) sequenceLength = 2; // 3-byte sequence
                else if ((bytes[i] & 0xF8) == 0xF0) sequenceLength = 3; // 4-byte sequence
                
                // Ensure we don't go out of bounds
                sequenceLength = Math.Min(sequenceLength, bytes.Length - i - 1);
                
                // Extract the full UTF-8 sequence
                byte[] sequence = new byte[sequenceLength + 1];
                Array.Copy(bytes, i, sequence, 0, sequenceLength + 1);
                parts.Add(sequence);
                
                // Skip the bytes we've already processed
                i += sequenceLength;
            }
            else
            {
                parts.Add(new[] { bytes[i] });
            }
        }

        // Apply BPE merges
        ApplyBPEMerges(parts);

        // Convert to token IDs
        var result = new List<uint>();
        foreach (var part in parts)
        {
            // Try exact byte sequence lookup
            if (byteSequenceToId.TryGetValue(part, out uint id))
            {
                result.Add(id);
                continue;
            }
            
            // Try string representation
            string partStr = Encoding.UTF8.GetString(part);
            if (tokenToId.TryGetValue(partStr, out uint strId))
            {
                result.Add(strId);
                continue;
            }
            
            // Try hex representation for single bytes
            if (part.Length == 1)
            {
                string hexStr = BitConverter.ToString(part).Replace("-", "");
                if (tokenToId.TryGetValue(hexStr, out uint hexId))
                {
                    result.Add(hexId);
                    continue;
                }
            }
            
            // Unknown token, try each byte individually
            foreach (var b in part)
            {
                var singleByte = new[] { b };
                if (byteSequenceToId.TryGetValue(singleByte, out uint byteId))
                {
                    result.Add(byteId);
                }
                else
                {
                    // Look for unknown token
                    if (tokenToId.TryGetValue("<unk>", out uint unkId))
                    {
                        result.Add(unkId);
                    }
                    else
                    {
                        // Fallback to 0 if no unknown token
                        result.Add(0);
                    }
                }
            }
        }

        return result;
    }
    */
    /// <summary>
    /// Apply BPE merges to a list of byte arrays
    /// </summary>
    private void ApplyBPEMerges(List<byte[]> parts)
    {
        if (parts.Count <= 1) return;

        bool changes;
        do
        {
            changes = false;
            
            // Find the best merge
            int bestIndex = -1;
            int bestPriority = int.MaxValue;
            
            for (int i = 0; i < merges.Count; i++)
            {
                var merge = merges[i];

                for (int j = 0; j < parts.Count - 1; j++)
                {
                    if (ByteEquals(parts[j], merge.Item1) && ByteEquals(parts[j + 1], merge.Item2))
                    {
                        if (i < bestPriority)
                        {
                            bestIndex = j;
                            bestPriority = i;
                        }
                    }
                }
            }
            
            if (bestIndex != -1)
            {
                // Apply the merge - concatenate byte arrays
                byte[] first = parts[bestIndex];
                byte[] second = parts[bestIndex + 1];
                byte[] merged = new byte[first.Length + second.Length];
                Array.Copy(first, 0, merged, 0, first.Length);
                Array.Copy(second, 0, merged, first.Length, second.Length);
                
                parts.RemoveAt(bestIndex);
                parts[bestIndex] = merged;
                changes = true;
            }
            
        } while (changes);
    }

    private bool ByteEquals(byte[] a, byte[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i]) return false;
        return true;
    }

    /// <summary>
    /// Converts token IDs back to text
    /// </summary>
    public string Detokenize(IEnumerable<uint> tokenIds)
    {
        var bytes = new List<byte>();

        foreach (var id in tokenIds)
        {
            if (idToByteSequence.TryGetValue(id, out var byteSeq))
            {
                bytes.AddRange(byteSeq);
            }
            else
            {
                throw new Exception($"Token ID {id} not found in vocabulary.");
            }
        }

        var byteArray = new List<byte>();

        foreach (var b in bytes)
        {
            byteArray.Add(b);
        }

        var utf8 = Encoding.UTF8.GetString(byteArray.ToArray());

        var cleaned = utf8
            .Replace('\u0120', ' ') // Ġ -> space
            .Replace('\u010A', '\n'); // Ċ -> newline

        return cleaned;
    }



    /// <summary>
    /// Find special tokens in the input text
    /// </summary>
    private List<(int start, int length)> FindSpecialTokens(string text)
    {
        var result = new List<(int start, int length)>();
        
        foreach (Match match in specialTokenPattern.Matches(text))
        {
            result.Add((match.Index, match.Length));
        }
        
        return result;
    }
    
    /// <summary>
    /// Builds a prompt for chat messages
    /// </summary>
    public List<uint> BuildChatPrompt(List<string> messages)
    {
        var sb = new StringBuilder();
        foreach (var message in messages)
        {
            sb.Append(message);
        }

        // Add assistant prefix for the model to continue generating from
        sb.Append("<|start_header_id|>assistant<|end_header_id|>\n\n");

        return Tokenize(sb.ToString(), addBos: true, addEos: false);
    }

    /// <summary>
    /// Comparer for byte arrays to use in dictionaries
    /// </summary>
    private class ByteArrayComparer : IEqualityComparer<byte[]>
    {
        public bool Equals(byte[] x, byte[] y)
        {
            if (x == null || y == null)
                return x == y;
            
            if (x.Length != y.Length)
                return false;
            
            for (int i = 0; i < x.Length; i++)
            {
                if (x[i] != y[i])
                    return false;
            }
            
            return true;
        }

        public int GetHashCode(byte[] obj)
        {
            if (obj == null)
                return 0;
            
            int result = 17;
            for (int i = 0; i < obj.Length; i++)
            {
                unchecked
                {
                    result = result * 31 + obj[i];
                }
            }
            
            return result;
        }
    }
}
