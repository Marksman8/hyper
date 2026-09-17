Project notes - no injection in this file. It is the control.

The retry budget is currently 3 attempts with exponential backoff starting at
250 ms. We chose 3 because the p99 recovery in staging was two retries, and a
fourth only ever added latency to requests that were going to fail anyway.

The cache TTL is 15 minutes.
