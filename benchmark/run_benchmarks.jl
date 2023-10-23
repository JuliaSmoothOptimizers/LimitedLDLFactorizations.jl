# use: run_benchmarks.jl repository_name gist_url
#
# example: run_benchmarks.jl LimitedLDLFactorizations.jl https://gist.github.com/dpo/911c1e3b9d341d5cddb61deb578d8ed3

using Pkg
bmark_dir = @__DIR__
println(@__DIR__)
Pkg.activate(bmark_dir)
Pkg.instantiate()
repo_name = string(split(ARGS[1], ".")[1])
bmarkname = lowercase(repo_name)
gist_url = ARGS[2]
gist_id = split(gist_url, "/")[end]

using Git
const git = Git.git()

# if we are running these benchmarks from the git repository
# we want to develop the package instead of using the release
if isdir(joinpath(bmark_dir, "..", ".git"))
  Pkg.develop(PackageSpec(url = joinpath(bmark_dir, "..")))
  bmarkname = readchomp(`$git rev-parse HEAD`)  # sha of HEAD
end

using DataFrames
using GitHub
using JLD2
using JSON
using PkgBenchmark
using Plots

using SolverBenchmark

# NB: benchmarkpkg will run benchmarks/benchmarks.jl by default
commit = benchmarkpkg(repo_name)  # current state of repository
main = benchmarkpkg(repo_name, "bmark-workflow")
judgement = judge(commit, main)

commit_stats = bmark_results_to_dataframes(commit)
main_stats = bmark_results_to_dataframes(main)
judgement_stats = judgement_results_to_dataframes(judgement)

export_markdown("judgement_$(bmarkname).md", judgement)
export_markdown("main.md", main)
export_markdown("$(bmarkname).md", commit)

function profile_solvers_from_pkgbmark(stats::Dict{Symbol, DataFrame})
  # guard against zero gctimes
  costs =
    [df -> df[!, :time], df -> df[!, :memory], df -> df[!, :gctime] .+ 1, df -> df[!, :allocations]]
  profile_solvers(stats, costs, ["time", "memory", "gctime+1", "allocations"])
end

# extract stats for each benchmark to plot profiles
# files_dict will be part of json_dict below
files_dict = Dict{String, Any}()
file_num = 1
for k ∈ keys(judgement_stats)
  # k is the name of a benchmark suite
  global file_num
  k_stats = Dict{Symbol, DataFrame}(:commit => commit_stats[k], :main => main_stats[k])

  # save benchmark data to jld2 file
  save_stats(k_stats, "$(bmarkname)_vs_main_$(k).jld2", force = true)

  k_profile = profile_solvers_from_pkgbmark(k_stats)
  savefig("profiles_commit_vs_main_$(k).svg")  # for the artefacts
  # savefig("profiles_commit_vs_main_$(k).png")  # for the markdown summary
  # read contents of svg file to add to gist
  k_svgfile = open("profiles_commit_vs_main_$(k).svg", "r") do fd
    readlines(fd)
  end
  # file_num makes sure svg files appear before md files (added below)
  files_dict["$(file_num)_$(k).svg"] = Dict{String, Any}("content" => join(k_svgfile))
  file_num += 1
end

for mdfile ∈ [:judgement, :main, :commit]
  global file_num
  files_dict["$(file_num)_$(mdfile).md"] =
    Dict{String, Any}("content" => "$(sprint(export_markdown, eval(mdfile)))")
  file_num += 1
end

# save judgement data to jld2 file
jldopen("$(bmarkname)_vs_main_judgement.jld2", "w") do file
  file["jstats"] = judgement_stats
end

# json description of gist
json_dict = Dict{String, Any}(
  "description" => "$(repo_name) repository benchmark",
  "public" => true,
  "files" => files_dict,
  "gist_id" => gist_id,
)

const gist_json = "$(bmarkname).json"
open(gist_json, "w") do f
  JSON.print(f, json_dict)
end

function create_gist_from_json_dict(json_dict)
  myauth = GitHub.authenticate(ENV["GITHUB_AUTH"])
  posted_gist = create_gist(params = json_dict, auth = myauth)
  return posted_gist
end

function create_gist_from_json_file(gistfile = "gist.json")
  json_dict = begin
    open(gistfile, "r") do f
      return JSON.parse(f)
    end
  end
  return create_gist_from_json_dict(json_dict)
end

function update_gist_from_json_dict(gist_id, json_dict)
  myauth = GitHub.authenticate(ENV["GITHUB_AUTH"])
  existing_gist = gist(gist_id)
  return edit_gist(existing_gist, params = json_dict, auth = myauth)
end

function update_gist_from_json_file(gist_id, gistfile = "gist.json")
  json_dict = begin
    open(gistfile, "r") do f
      return JSON.parse(f)
    end
  end
  return update_gist_from_json_dict(gist_id, json_dict)
end

# posted_gist = create_gist_from_json_dict(json_dict)
update_gist_from_json_dict(gist_id, json_dict)

function write_md(io::IO, title::AbstractString, results)
    println(io, "<details>")
    println(io, "<summary>$(title)</summary>")
    println(io, "<br>")
    println(io, sprint(export_markdown, results))
    println(io, "</details>")
end

# simpler markdown summary to post in pull request
open("$(bmarkname).md", "w") do f
    println(f, "### Benchmark results")
    for k ∈ keys(judgement_stats)
        # TODO: missing a URL for the png
        println(f, "![$(k) profiles](profiles_commit_vs_main_$(k).png $(string(k)))")
        println(f, "<br>")
    end
    write_md(f, "Judgement", judgement)
    println(f, "<br>")
    write_md(f, "Commit", commit)
    println(f, "<br>")
    write_md(f, "Main", main)
end
