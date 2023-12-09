---
title: "Building an image search engine using CLIP"
date: 2023-10-21
# weight: 1
# aliases: ["/first"]
tags: ["openai", "clip", "iOS", "Swift", "CoreML"]
author: "AJ"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: true
hidemeta: false
comments: false
description: ""
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: true
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: https://cdn.araintelligence.com/blog-posts/clip.webp
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
---

![Clip Hero Image](https://cdn.araintelligence.com/blog-posts/clip.webp)

# A quick introduction to CLIP

[CLIP](https://openai.com/research/clip) is a neural network architecture developed by OpenAI that learns to condition images on natural languages.
By not directly optmising between images and class labels, the CLIP model is more robust due to the constraints imposed on images by natural human language supervision.

A rough overview is to view its training process as tying a link between some representation (embedding) of the current image to the embedding of a text description of the image.
The core difference is the shift from classification labels (think ImageNet labels e.g "cat", "dog" etc) to text embeddings.
![CLIP inference overview](https://cdn.araintelligence.com/blog-posts/CLIP/clip-inference.png)