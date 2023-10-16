import * as http from 'node:http';

interface StreamingRequestOptions<T> {
  method: 'GET' | 'POST';
  headers?: Record<string, string>;
  timeout?: number;
  body?: T;
  signal?: AbortSignal;
}

function toNodeRequestOptions<T>(url: string, options: StreamingRequestOptions<T>): http.RequestOptions {
  const requestUrl = new URL(url);

  return {
    protocol: requestUrl.protocol,
    hostname: requestUrl.hostname,
    port: requestUrl.port,
    path: requestUrl.pathname,
    method: options.method,
    headers: options.headers,
    timeout: options.timeout,
  };
}

enum StatusCode {
  OK = 200,
}

export function streamingRequest<T, R>(
  url: string,
  onDataChunk: (chunk: R) => void,
  options: StreamingRequestOptions<T>
) {
  const requestOptions = toNodeRequestOptions(url, options);
  let error: Error | null = null;

  return new Promise<void>((resolve, reject) => {
    const request = http.request(requestOptions, (response) => {
      if (response.statusCode !== StatusCode.OK) {
        reject(response.statusMessage);
        return;
      }

      response.setEncoding('utf8');
      response.on('data', onDataChunk);
      response.once('error', (err) => (error = err));
      response.once('close', () => response.removeAllListeners());
    });

    request.once('error', (err) => (error = err));

    request.once('close', () => {
      if (error) {
        reject(error);
        return;
      }
      resolve();
    });

    request.write(JSON.stringify(options.body));
    request.end();

    if (options.signal) {
      options.signal.onabort = () => request.destroy();
    }
  });
}
