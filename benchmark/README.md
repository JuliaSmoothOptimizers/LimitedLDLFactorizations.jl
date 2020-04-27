# Benchmark

## Benchmark datasets

The [sqd-collection](https://github.com/optimizers/sqd-collection) repository contains 351 symmetric and quasi-definite linear systems.

```bash
git clone https://github.com/optimizers/sqd-collection.git
```

## Running the benchmark

Then, run the benchmark as follows:
```julia
using PkgBenchmark
import LimitedLDLFactorizations
results = benchmarkpkg("LimitedLDLFactorizations")
export_markdown("results.md", results)
```

## Comparing two commits

To compare against the `master` branch
```julia
using PkgBenchmark
import LimitedLDLFactorizations
judgement = judge("LimitedLDLFactorizations", "master")
export_markdown("judgement.md", judgement)
```

## Profiling BenchmarkResults and BenchmarkJudgement

```julia
using SolverBenchmark
p_results   = profile_solvers(results)
p_judgement = profile_package(judgement)
```

## Uploading benchmark results and performance profiles

The environment variable **GITHUB_AUTH** (ENV["GITHUB_AUTH"] in Julia) must contain your personal access token that allows to create gists.

```julia
using GitHub, JSON, PkgBenchmark
markdowncontent = escape_string(read(markdownname, String))
gist_json = JSON.parse(
            """
            {
              "description": "A benchmark for LimitedLDLFactorization.jl",
              "public": true,
              "files": {
                "benchmark.md": {
                "content": "$(markdowncontent)"
                }
              }
            }
            """)
myauth = authenticate(ENV["GITHUB_AUTH"])
posted_gist = create_gist(params = gist_json, auth = myauth)
println(posted_gist.html_url)
```

```julia
using SolverBenchmark
to_gist(results, p_results)
to_gist(judgement, p_judgement)
```
