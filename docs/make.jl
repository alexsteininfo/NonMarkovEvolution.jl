using Documenter
using NonMarkovEvolution

DocMeta.setdocmeta!(NonMarkovEvolution, :DocTestSetup,
                    :(using NonMarkovEvolution); recursive = true)

makedocs(
    sitename = "NonMarkovEvolution.jl",
    modules = [NonMarkovEvolution],
    authors = "Alexander Stein",
    pages = [
        "Home" => "index.md",
        "Concepts" => "concepts.md",
        "The simulation block" => "blocks.md",
        "Mutations and selection" => "selection.md",
        "Output" => "output.md",
        "Tree statistics" => "statistics.md",
        "Sampling" => "sampling.md",
        "Performance" => "performance.md",
        "Limitations and open questions" => "limitations.md",
        "API reference" => "api.md",
    ],
    doctest = true,
    checkdocs = :exports,
    format = Documenter.HTML(prettyurls = get(ENV, "CI", "false") == "true"),
)

deploydocs(
    repo = "github.com/alexsteininfo/NonMarkovEvolution.jl.git",
    devbranch = "main",
)
