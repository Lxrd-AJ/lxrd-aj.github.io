# Common Commands

### Creating a new post workflow
Command hugo to create a new markdown file
```
hugo new --kind post content/blog/ml/my-new-ml-post.md
```

Use `draft: true` in the yaml header to prevent the draft page from being published live.

### Development server
To run the dev server, run
```
hugo server --buildDrafts
```

## Other Resources
* Configuring the `hugo-PaperMod` theme 
    * https://github.com/adityatelange/hugo-PaperMod/wiki/FAQs 
    * https://github.com/adityatelange/hugo-PaperMod/wiki/Features 