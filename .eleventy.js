module.exports = function (eleventyConfig) {
  eleventyConfig.addPassthroughCopy({ "src/assets": "assets" });
  eleventyConfig.addPassthroughCopy({ "src/posts/synthetic-data-selection/assets": "posts/synthetic-data-selection/assets" });
  eleventyConfig.addPassthroughCopy({ "src/posts/robust-distilled-datasets/assets": "posts/robust-distilled-datasets/assets" });
  eleventyConfig.addPassthroughCopy({ "node_modules/katex/dist/katex.min.css": "assets/katex.min.css" });
  eleventyConfig.addPassthroughCopy({ "node_modules/katex/dist/fonts": "assets/fonts" });
  eleventyConfig.addPassthroughCopy({ "node_modules/katex/dist/katex.min.js": "assets/katex.min.js" });
  eleventyConfig.addPassthroughCopy({ "node_modules/katex/dist/contrib/auto-render.min.js": "assets/auto-render.min.js" });
  eleventyConfig.addFilter("readableDate", (date) => new Intl.DateTimeFormat("zh-CN", { timeZone: "UTC", year: "numeric", month: "2-digit", day: "2-digit" }).format(date).replaceAll("/", "."));
  eleventyConfig.addFilter("htmlDate", (date) => date.toISOString().slice(0, 10));
  eleventyConfig.addFilter("limit", (items, count) => items.slice(0, count));
  eleventyConfig.addFilter("topicSlug", (topic) => encodeURIComponent(topic));
  eleventyConfig.addFilter("filterByTopic", (posts, topic) => posts.filter((post) => (post.data.topics || []).includes(topic)));

  eleventyConfig.addCollection("posts", (api) => api.getFilteredByGlob("src/posts/**/*.md").sort((a, b) => b.date - a.date));
  eleventyConfig.addCollection("topicNames", (api) => {
    const topics = new Set();
    api.getFilteredByGlob("src/posts/**/*.md").forEach((item) => (item.data.topics || []).forEach((topic) => topics.add(topic)));
    return [...topics].sort((a, b) => a.localeCompare(b, "zh-CN"));
  });

  return {
    dir: { input: "src", includes: "_includes", output: "_site" },
    templateFormats: ["njk", "md"],
    markdownTemplateEngine: "njk",
    htmlTemplateEngine: "njk",
    pathPrefix: "/"
  };
};
