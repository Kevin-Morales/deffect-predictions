
# Compiles the R Markdown file to HTML.
 
library(rmarkdown)

render("final-draft.Rmd", c("html_document", "pdf_document"))