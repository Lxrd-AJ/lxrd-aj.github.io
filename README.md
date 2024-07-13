# Common Commands

## Creating a new post workflow

### Convert ipynb notebook to markdown
Ensure `pandoc` installed. See https://github.com/jupyter/nbconvert for more info.
Usually this is as trivial as running `brew install pandoc`

To convert the notebook run `jupyter nbconvert --to markdown mynotebook.ipynb`

### Manually create a new hugo markdown post OR **Copy yaml header from existing post**
Command hugo to create a new markdown file
```
hugo new --kind post content/blog/ml/my-new-ml-post.md
```

Use `draft: true` in the yaml header to prevent the draft page from being published live.

### Development server for Preview
To run the dev server, run
```
hugo server --buildDrafts
```

### Deploy

## Other Resources
* Configuring the `hugo-PaperMod` theme 
    * https://github.com/adityatelange/hugo-PaperMod/wiki/FAQs 
    * https://github.com/adityatelange/hugo-PaperMod/wiki/Features 