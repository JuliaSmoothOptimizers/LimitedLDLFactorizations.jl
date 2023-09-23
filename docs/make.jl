using Documenter, LimitedLDLFactorizations

makedocs(
  modules = [LimitedLDLFactorizations],
  doctest = true,
  linkcheck = true,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "LimitedLDLFactorizations.jl",
  pages = Any["Home" => "index.md", "Tutorial" => "tutorial.md", "Reference" => "reference.md"],
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/LimitedLDLFactorizations.jl.git",
  devbranch = "main",
)
