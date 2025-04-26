using Microsoft.Extensions.AI;

namespace NFAI.Models;

public class GenericChatClient(IInferenceProvider inferenceProvider) : IChatClient
{
    public void Dispose()
    {
        GC.SuppressFinalize(this);
        inferenceProvider.Dispose();
    }

    public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        return inferenceProvider.GetStreamingResponseAsync(messages, options, cancellationToken).ToChatResponseAsync(cancellationToken);
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        throw new NotImplementedException();
    }

    public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        return inferenceProvider.GetStreamingResponseAsync(messages, options, cancellationToken);
    }
}
