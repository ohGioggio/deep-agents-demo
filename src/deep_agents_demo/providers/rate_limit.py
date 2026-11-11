"""One request-rate cap shared by the chat model and the embeddings."""


_limiters = {}


def rate_limiter(requests_per_minute: float):
    """One shared limiter per rate, or None if unlimited.

    The cap covers chat and embeddings together, because a provider's limit
    (NVIDIA's free tier: about 40 per minute) usually applies to the API key,
    not to each kind of request. The same rate always returns the same limiter.
    """
    if requests_per_minute <= 0:
        return None
    if requests_per_minute not in _limiters:
        from langchain_core.rate_limiters import InMemoryRateLimiter
        _limiters[requests_per_minute] = InMemoryRateLimiter(
            requests_per_second=requests_per_minute / 60, check_every_n_seconds=0.1, max_bucket_size=1)
    return _limiters[requests_per_minute]
