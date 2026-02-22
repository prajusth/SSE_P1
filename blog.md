# Measuring the Energy Footprint of Compression: Comparing 7zip, gzip, and zstd

## Introduction

Data centers now consume roughly 1 to 2% of global electricity, and software runs on billions of devices worldwide. Understanding the energy consumption of everyday software operations has never been more important. While much attention focuses on large scale systems like machine learning training or blockchain networks, even mundane operations like file compression contribute to our collective digital carbon footprint.

This study investigates a deceptively simple question: **How much energy does file compression actually consume, and does your choice of compression tool matter?**

Compression is everywhere in modern computing. Every time you download a software package, back up files to the cloud, or archive logs on a server, compression algorithms are at work. Yet developers rarely consider the energy implications when choosing between tools like gzip, 7zip, or the newer zstd. We set out to measure these differences systematically.

Our experiment compares three popular compression tools (7zip, gzip, and zstd) across different file types and sizes, measuring actual energy consumption using hardware level instrumentation. The results reveal surprising insights about the energy efficiency tradeoffs in compression software.

## Background: Why Energy Measurement Matters

### The Rise of Green Software Engineering

Software sustainability has emerged as a critical concern in computer science. The Green Software Foundation and initiatives like the Software Carbon Intensity (SCI) specification reflect growing awareness that software design choices have real environmental consequences. Energy consumption directly translates to carbon emissions, with the relationship depending on the local power grid's carbon intensity.

Energy is measured in Joules (J), while power (the rate of energy consumption) is measured in Watts (W). The relationship is straightforward: Energy = Power × Time, or equivalently, 1 Watt = 1 Joule per second. When measuring software energy consumption, we typically sample power readings over time and integrate them to obtain total energy.

### The Energy Performance Tradeoff

A key insight from green software research is that faster is not always greener. Consider two approaches to a task:
- **Approach A**: Completes in 2 seconds at 50W average power = 100J
- **Approach B**: Completes in 4 seconds at 20W average power = 80J

Despite being twice as slow, Approach B consumes 20% less energy. This counterintuitive relationship is captured by the **Energy Delay Product (EDP)**, calculated as EDP = Energy × Time. EDP helps identify solutions that balance both energy efficiency and performance.

For compression specifically, this raises interesting questions. Fast compression modes complete quickly but may consume more instantaneous power. Thorough compression takes longer but achieves better ratios. Which approach is actually more energy efficient?

## Methodology

### Experimental Setup

We designed a controlled experiment to measure energy consumption across multiple dimensions.

**Hardware Environment:**
All experiments were conducted on an HP laptop running dual boot Linux. Energy measurements were collected using RAPL (Running Average Power Limit) counters via EnergiBridge. Results may vary on different hardware configurations, and the absolute energy values are specific to this machine. However, the relative comparisons between tools should generalize across similar x86 systems.

**Compression Tools Tested:**
- **7zip**: A popular open source archiver known for high compression ratios
- **gzip**: The ubiquitous GNU compression utility, standard on Unix systems
- **zstd (Zstandard)**: Facebook's modern compression algorithm, designed for speed

**Compression Levels:**
- **Fast**: Prioritizes speed over compression ratio (7z mx=1, gzip 1, zstd 1)
- **Default**: Balanced setting as recommended by each tool (7z mx=5, gzip 6, zstd 3)

**File Types:**
- **Text logs (.log)**: Simulated server logs, highly repetitive and very compressible
- **CSV data (.csv)**: Tabular datasets with structured data and repeated patterns
- **PDF documents (.pdf)**: Already internally compressed, minimal external compression benefit
- **BMP images (.bmp)**: Raw bitmap data, uncompressed and highly compressible

**File Sizes:**
- Small: 10 MB (typical email attachment)
- Medium: 100 MB (software package or dataset)
- Large: 500 MB (video file or database dump)

This design yields 3 × 2 × 4 × 3 = 72 unique configurations, providing comprehensive coverage of real world compression scenarios.

### Energy Measurement

We used **EnergiBridge**, a cross platform tool for measuring software energy consumption. EnergiBridge interfaces with hardware energy counters, specifically Intel's RAPL (Running Average Power Limit) interface or AMD's equivalent, to capture actual energy usage at the CPU level.

The measurement process works as follows:
1. EnergiBridge starts recording energy counter values
2. The compression command executes
3. EnergiBridge captures the final counter values
4. Energy consumption = final reading − initial reading

For power based measurements (common on some platforms), we apply the **trapezoid rule** to integrate power samples over time:

```
Energy ≈ Σ [(P[i] + P[i+1]) / 2 × (t[i+1] - t[i])]
```

This numerical integration accounts for varying power consumption throughout execution.

### Ensuring Reproducibility

Following established best practices for energy measurement experiments, we implemented several controls:

1. **Zen Mode**: Before experiments, we kill background processes (browsers, media players, chat applications) that could introduce measurement noise.

2. **CPU Warmup**: A five minute CPU warmup period (computing Fibonacci numbers) ensures the processor reaches steady state thermal conditions before measurements begin.

3. **Randomized Execution Order**: Each round randomizes the order of test configurations to prevent time based bias (for example, thermal throttling affecting later runs).

4. **Multiple Repetitions**: We perform 30 measured repetitions of each configuration, plus 5 warmup runs that are discarded. This provides statistical power for reliable comparisons.

5. **Cooldown Periods**: A minimum ten second cooldown between runs (or 50% of the run time, whichever is greater) prevents tail energy from one run bleeding into the next measurement.

6. **RAPL Overflow Handling**: AMD CPUs' energy counters wrap around at 2^16 Joules. Our code detects and corrects for this overflow.

### Metrics Collected

For each run, we record:
- **Energy (Joules)**: Total CPU energy consumed during compression
- **Wall Time (seconds)**: Total execution time
- **Compression Ratio**: Output size / Input size (lower is better)
- **Energy per MB**: Energy normalized by input size, enabling fair comparison across file sizes

## Results

Our experiment completed 2,520 compression runs (72 configurations × 35 rounds), with 2,160 measured runs after excluding warmup iterations.

### Overall Energy Consumption by Tool

The results reveal dramatic differences between compression tools:

| Tool   | Level   | Mean Energy (J) | Std Dev (J) | Mean Time (s) | Compression Ratio |
|--------|---------|-----------------|-------------|---------------|-------------------|
| 7zip   | fast    | 52.00           | 64.83       | 3.32          | 0.44              |
| 7zip   | default | 221.97          | 204.87      | 16.22         | 0.39              |
| gzip   | fast    | 40.34           | 51.21       | 5.33          | 0.52              |
| gzip   | default | 66.32           | 69.02       | 8.45          | 0.51              |
| zstd   | fast    | 7.53            | 7.56        | 1.31          | 0.54              |
| zstd   | default | 15.15           | 17.12       | 2.16          | 0.53              |

**Key observation**: zstd consumes **12x less energy** than 7zip on average (11.34 J vs 136.99 J), while gzip falls in between at 53.33 J. This represents a substantial difference that would compound significantly at scale.

### Energy Scaling with File Size

Energy consumption scales with file size, but the scaling factor varies dramatically by tool:

| Tool   | Small (10 MB) | Medium (100 MB) | Large (500 MB) |
|--------|---------------|-----------------|----------------|
| 7zip   | 11.58 J       | 93.92 J         | 305.46 J       |
| gzip   | 3.03 J        | 27.03 J         | 129.94 J       |
| zstd   | 1.06 J        | 5.72 J          | 27.23 J        |

When normalized to energy per megabyte:
- **zstd fast**: 0.053 J/MB
- **zstd default**: 0.084 J/MB
- **gzip fast**: 0.193 J/MB
- **gzip default**: 0.330 J/MB
- **7zip fast**: 0.261 J/MB
- **7zip default**: 1.473 J/MB

zstd is remarkably consistent, consuming approximately **0.05 to 0.08 J per MB** regardless of file size, while 7zip at default settings consumes nearly **1.5 J per MB**, a 20x difference.

### Impact of File Type

Energy consumption varies by file type, reflecting the different compressibility characteristics:

| Tool   | CSV      | Image    | PDF      | Text     |
|--------|----------|----------|----------|----------|
| 7zip   | 133.82 J | 111.91 J | 142.11 J | 160.10 J |
| gzip   | 40.55 J  | 63.78 J  | 69.47 J  | 39.53 J  |
| zstd   | 8.51 J   | 14.04 J  | 14.27 J  | 8.52 J   |

Interestingly, PDFs (which are already internally compressed) still consume significant energy despite achieving poor compression ratios. This suggests that **detecting precompressed content and skipping compression could yield energy savings** without sacrificing functionality.

Text and CSV files, both highly compressible, show the best energy efficiency across all tools, with zstd consuming only around 8.5 J on average.

### The Speed Energy Tradeoff

Comparing fast vs default compression levels reveals a clear pattern:

| Tool   | Fast Energy | Default Energy | Increase | Fast Time | Default Time |
|--------|-------------|----------------|----------|-----------|--------------|
| 7zip   | 52.00 J     | 221.97 J       | +327%    | 3.32 s    | 16.22 s      |
| gzip   | 40.34 J     | 66.32 J        | +64%     | 5.33 s    | 8.45 s       |
| zstd   | 7.53 J      | 15.15 J        | +101%    | 1.31 s    | 2.16 s       |

**Finding**: Fast modes consistently use less energy. The default settings, while achieving marginally better compression ratios (0.39 vs 0.44 for 7zip), consume 64 to 327% more energy. For most use cases, **fast compression is both faster and greener**.

The compression ratio improvement from fast to default is modest:
- 7zip: 0.44 → 0.39 (11% better compression, 327% more energy)
- gzip: 0.52 → 0.51 (2% better compression, 64% more energy)
- zstd: 0.54 → 0.53 (2% better compression, 101% more energy)

### Statistical Analysis

To ensure our findings are statistically robust, we performed comprehensive hypothesis testing on the energy consumption data.

**Normality Testing**

We first assessed whether energy measurements followed a normal distribution using the Shapiro Wilk test (α = 0.05) for each of the 72 experimental configurations:

- **55 of 72 groups (76.4%)** passed the normality test (p > 0.05)
- Non normal distributions were observed primarily in small file sizes and some fast compression modes
- Skewness values ranged from 0.88 to +2.64, with most groups showing mild positive skew

Given the mixed normality results, we employed both parametric (Welch's t test) and non parametric (Mann Whitney U) tests depending on each comparison's distribution characteristics.

**Provider Comparisons**

We tested whether energy consumption differs significantly between compression tools across all 72 configurations (24 unique file/size combinations × 3 pairwise comparisons):

| Comparison       | Significant Tests | Mean Effect Size (Cohen's d) | Interpretation |
|-----------------|-------------------|------------------------------|----------------|
| 7zip vs gzip    | 24/24 (100%)      | 117.2                        | Very large     |
| 7zip vs zstd    | 24/24 (100%)      | 172.8                        | Very large     |
| gzip vs zstd    | 24/24 (100%)      | 191.7                        | Very large     |

**All 72 provider comparisons were statistically significant (p < 0.001)**. The effect sizes are extraordinarily large. Cohen's d values exceeding 100 indicate virtually no overlap between distributions. For context, a Cohen's d of 0.8 is typically considered large in social sciences; our measurements show effects 100 to 200 times that threshold.

**Compression Level Comparisons**

We tested whether fast vs default compression levels differ in energy consumption for each tool:

| Provider | Significant Comparisons | Mean % Increase (fast → default) |
|----------|------------------------|----------------------------------|
| 7zip     | 12/12 (100%)           | +665%                            |
| gzip     | 12/12 (100%)           | +106%                            |
| zstd     | 12/12 (100%)           | +76%                             |

**All 36 level comparisons were statistically significant (p < 0.001)**. Default compression modes consistently consume more energy than fast modes, with 7zip showing the most dramatic difference (up to 1532% more energy for text files at medium size).

**Summary of Statistical Findings**

The statistical analysis strongly supports our conclusions:
1. Tool differences are not due to random variation. They represent genuine, large magnitude effects
2. The choice between fast and default modes has measurable, significant energy implications
3. These effects are consistent across file types and sizes, strengthening generalizability

## Discussion

### Key Findings

**Finding 1: Tool choice matters dramatically, up to 12x difference**

The most striking result is the magnitude of difference between tools. zstd consumes 12x less energy than 7zip and 4.7x less than gzip for equivalent tasks. This far exceeds what we might expect from similar compression tools. The difference stems from zstd's modern algorithm design, which prioritizes speed and efficiency over maximum compression ratio.

**Finding 2: Fast modes are genuinely energy efficient**

Contrary to the intuition that faster means more power hungry, fast compression modes consistently consume less total energy. 7zip's default mode uses 327% more energy than its fast mode for only 11% better compression. This confirms that for energy efficiency, completing work quickly at moderate power beats sustained high effort compression.

**Finding 3: Precompressed files waste energy**

PDFs showed the highest energy consumption relative to compression benefit across all tools. Since PDFs use internal compression (FlateDecode), external compression provides minimal size reduction while still consuming full processing energy. Intelligent compression pipelines should detect and skip already compressed formats.

### Practical Implications

**For System Administrators:**
- Consider energy consumption when selecting compression tools for backup systems and log rotation
- Implement file type detection to avoid wasteful compression of already compressed data
- For battery powered or energy constrained environments, tool selection matters

**For Developers:**
- Default compression settings may not be optimal for energy efficiency
- When packaging software or assets, test different tools' energy profiles
- Consider exposing compression choices to users with energy conscious defaults

**For Cloud Providers:**
- At scale, small per operation energy differences multiply into significant costs
- Compression energy should factor into total cost of ownership calculations
- Providing energy efficient compression as a service feature could differentiate offerings

### Limitations

Several factors limit the generalizability of our findings:

1. **Single machine measurements**: Results may vary on different hardware (Intel vs AMD, different generations, ARM processors)

2. **CPU only energy**: We measure CPU energy consumption only. Storage I/O, memory, and network energy are not captured.

3. **Synthetic test files**: While designed to represent real world scenarios, generated test files may not perfectly reflect production workloads.

4. **Limited compression levels**: Testing only fast and default leaves out maximum compression settings used in archival scenarios.

### Future Work

Several directions could extend this research:

- **Cross platform comparison**: Measure the same tools on Windows, macOS, and different Linux distributions
- **Hardware diversity**: Test on ARM processors (Raspberry Pi, Apple Silicon, cloud ARM instances)
- **Decompression energy**: Compression is only half the story. Decompression energy matters too
- **End to end scenarios**: Measure compression + network transfer + decompression for cloud backup use cases
- **Real workload characterization**: Profile actual production compression workloads to validate synthetic benchmarks

## Conclusion

This study demonstrates that compression tool selection has dramatic energy implications, far greater than most developers would expect. Our measurements show that **zstd consumes 12x less energy than 7zip** and nearly **5x less than gzip** for equivalent compression tasks.

The practical recommendations are clear:
1. **Use zstd** when energy efficiency matters. It's fastest, most efficient, and provides reasonable compression
2. **Prefer fast modes** over default or maximum settings. The energy cost of marginal compression improvements is substantial
3. **Skip compression for precompressed formats** like PDF, JPEG, or ZIP files. You waste energy for negligible benefit

At scale, these differences compound dramatically. Consider a backup system processing 1 TB of data daily:
- Using 7zip default: ~1,500 kJ/day
- Using zstd fast: ~50 kJ/day
- **Annual savings: ~520 MJ ≈ 144 kWh**

For a data center, multiply by thousands of servers. The choice of compression tool becomes a meaningful factor in both energy costs and carbon footprint.

More broadly, this work illustrates why green software engineering matters. Everyday operations that seem trivial, like compressing a file, have measurable environmental impact when performed at scale. By measuring and optimizing these operations, we can build more sustainable software systems without sacrificing functionality.

## Reproducibility

All code, data, and analysis scripts are available in our replication package:
- **Repository**: https://github.com/prajusth/SSE_P1
- **Test data generation**: `python3 scripts/generate_test_data.py`
- **Experiment execution**: `python3 scripts/run_experiment.py`
- **Analysis**: `python3 analysis/analyze.py`

The experiment is designed to run on Linux systems with EnergiBridge installed. Approximate runtime is 4 to 6 hours for the full 30 repetition experiment.

## References

1. Green Software Foundation. "Software Carbon Intensity (SCI) Specification." https://greensoftware.foundation/
2. EnergiBridge documentation and source code
3. Intel RAPL documentation for energy measurement
4. Pereira, R., et al. "Energy Efficiency across Programming Languages." SLE 2017.
5. Pinto, G., Castor, F. "Energy Efficiency: A New Concern for Application Software Developers." CACM 2017.

---

*This blog post documents Project 1 of the Sustainable Software Engineering course, focusing on measuring and comparing energy consumption in common software use cases.*
