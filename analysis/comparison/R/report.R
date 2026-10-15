# Markdown reports for simulation phases. The same renderer serves the Gate C1
# synthetic render, the Gate D calibration report, and later phases.

cmp_fmt <- function(x, digits = 4) {
  ifelse(is.na(x), "NA", formatC(x, digits = digits, format = "f"))
}

cmp_md_table <- function(df) {
  header <- paste0("| ", paste(names(df), collapse = " | "), " |")
  rule <- paste0("|", paste(rep("---", ncol(df)), collapse = "|"), "|")
  body <- apply(df, 1L, function(r) paste0("| ", paste(r, collapse = " | "), " |"))
  c(header, rule, body)
}

cmp_summary_table <- function(summary) {
  data.frame(
    method = summary$method,
    risk = formatC(summary$family_balanced_risk, digits = 3, format = "e"),
    ratio = cmp_fmt(summary$ratio),
    ratio_95 = ifelse(summary$method == "additive", "denominator",
                      paste0("[", cmp_fmt(summary$ratio_lower_95), ", ",
                             cmp_fmt(summary$ratio_upper_95), "]")),
    class = summary$classification,
    gain_recovered = cmp_fmt(summary$gain_recovered, 3),
    coverage = cmp_fmt(summary$pointwise_coverage, 3),
    simultaneous = cmp_fmt(summary$simultaneous_coverage, 3),
    max_std_bias = cmp_fmt(summary$max_abs_standardized_bias, 3),
    failures = cmp_fmt(summary$job_failure_rate, 3),
    stringsAsFactors = FALSE
  )
}

# sections: list of list(title, summary (cmp_cell_summary output), note)
cmp_render_report <- function(path, title, preamble, sections, session = NULL) {
  lines <- c(paste("#", title), "", preamble, "")
  for (s in sections) {
    lines <- c(lines, paste("##", s$title), "")
    if (!is.null(s$note)) lines <- c(lines, s$note, "")
    if (!is.null(s$summary)) {
      lines <- c(lines,
                 sprintf("Replications used: %d of %d. Oracle/additive risk: %s.",
                         s$summary$replications_used, s$summary$replications_total,
                         cmp_fmt(s$summary$headroom)), "",
                 cmp_md_table(cmp_summary_table(s$summary$summary)), "")
    }
    if (!is.null(s$table)) lines <- c(lines, cmp_md_table(s$table), "")
  }
  if (!is.null(session)) {
    lines <- c(lines, "## Session", "", "```yaml", yaml::as.yaml(session), "```", "")
  }
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  writeLines(lines, path)
  invisible(path)
}
