# Silas · AI & Agent

这是 Silas 的中文论文阅读与研究札记站点，记录 AI、Agent 以及相关工程实践。

站点使用 Eleventy 将 Markdown 编译成静态 HTML，通过 GitHub Actions 自动部署到 GitHub Pages。

## 本地运行

需要 Node.js 22 或更高版本。

```bash
npm install
npm run start
```

然后打开 <http://localhost:4173/>。修改 `src/` 下的文件后，开发服务器会自动重新构建。

正式构建：

```bash
npm run build
```

生成目录是 `_site/`，它是构建产物，不需要提交到 Git。

## 新增文章

在 `src/posts/` 下新建一个 Markdown 文件。文件名建议使用日期加英文短名：

```text
src/posts/2026-10-01-agent-memory.md
```

文件开头添加 front matter：

```yaml
---
title: "文章标题"
description: "文章摘要，会显示在首页和 Archive。"
date: 2026-10-01
topics:
  - Agent
  - 大语言模型
tags:
  - 阅读笔记
layout: post.njk
---
```

然后在 front matter 后直接写 Markdown 正文：

```markdown
## 小节标题

这里是正文，可以使用**加粗**、列表、引用和代码。

````python
print("hello")
````
```

文章日期决定排序顺序。首页只显示最近 5 篇，全部文章可以在 `/archive/` 查看。

## 专题

专题由文章的 `topics` 字段自动生成，不需要单独维护专题列表：

```yaml
topics:
  - Agent
  - 论文阅读
```

多个文章使用同一个专题名称时，它们会自动出现在该专题下。专题名称需要保持完全一致，例如 `Agent` 和 `agent` 会被视为两个专题。

## 公式

支持 KaTeX 公式。行内公式使用单个美元符号：

```markdown
这是一个行内公式 $E = mc^2$。
```

块级公式使用两个美元符号：

```markdown
$$
P(y \mid x) = \frac{\exp(f(x,y))}{\sum_{y'} \exp(f(x,y'))}
$$
```

公式在浏览器端渲染，使用 KaTeX 官方资源。当前站点暂时保持浅色模式，避免图片、表格和公式在深色主题下出现额外兼容问题。

## 图片

推荐把文章图片放在 `src/assets/images/`：

```text
src/assets/images/agent-architecture.png
```

Markdown 中这样引用：

```markdown
![Agent 架构图](/assets/images/agent-architecture.png)
```

图片会被复制到构建结果中。文件名建议使用英文、数字和连字符，避免空格和特殊字符。

## 表格

Markdown 表格可以直接使用：

```markdown
| 方法 | 优点 | 局限 |
| --- | --- | --- |
| 方法 A | 简单 | 表达能力有限 |
| 方法 B | 灵活 | 配置更复杂 |
```

## 发布到 GitHub

确认本地预览和构建无误后：

```bash
git status
git add .
git commit -m "Add new article"
git push origin main
```

推送后，GitHub Actions 会自动执行构建和部署。可以在仓库的 **Actions** 页面查看进度，部署完成后访问：

<https://realnghon.github.io/>

## 目录说明

```text
src/posts/              文章 Markdown
src/_includes/          页面模板
src/assets/site.css     站点样式
src/assets/images/      文章图片
src/index.njk           首页
src/archive.njk         全部文章页
src/topics/index.njk    专题页
src/about.njk           关于页
.eleventy.js            构建配置
.github/workflows/      GitHub Pages 部署流程
```
