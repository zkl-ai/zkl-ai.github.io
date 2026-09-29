import { getCollection } from 'astro:content';

export const PAGE_SIZE = 10;

export async function getPublishedPosts() {
  return (await getCollection('blog'))
    .filter((p) => !p.data.draft)
    .sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());
}
