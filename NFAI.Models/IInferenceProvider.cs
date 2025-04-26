using Microsoft.Extensions.AI;

namespace NFAI.Models;

public interface IInferenceProvider : IDisposable
{
    /// <summary>
    /// Gets the model name.
    /// </summary>
    string ModelName { get; init; }

    public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default);
}
