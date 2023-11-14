import { LRUCache } from 'lru-cache';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const CACHES = [] as LRUCache<any, any>[];

export const clearLruCache = (): void => {
  for (const cache of CACHES) {
    cache.clear();
  }
};

type AsyncMethodType<R> = (this: unknown, ...args: unknown[]) => Promise<R>;

interface ILRUCacheConfig<R> {
  skipAddToCache?: (result: R) => boolean;
}

export function lruCache<V extends object, K extends string = string>(
  config?: ILRUCacheConfig<V | null>
): (target: unknown, propertyKey: string, descriptor: PropertyDescriptor) => PropertyDescriptor {
  return (target: unknown, propertyKey: string, descriptor: PropertyDescriptor) => {
    const originalMethod = descriptor.value as AsyncMethodType<V>;

    const cache = new LRUCache<K, V>({ max: 100 });
    CACHES.push(cache);

    descriptor.value = async function (data: object): Promise<V | null> {
      const key = JSON.stringify(data) as K;

      if (cache.has(key)) {
        return cache.get(key)!;
      }

      const result: V | null = await originalMethod.apply(this, [data]);

      if (result && !config?.skipAddToCache?.(result)) {
        cache.set(key, result);
      }

      return result;
    };

    return descriptor;
  };
}
