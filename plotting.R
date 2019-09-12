require(gridExtra)
require(tidyverse)
require(grid)
require(xtable)


ggmat <- tibble(values = c(c(0, beta),
                           as.vector(coef(cv.l, s = cv.l$lambda.1se)),
                           as.vector(coef(cv.r, s = cv.r$lambda.1se)),
                           as.vector(coef(cv.ropt, s = cv.ropt$lambda.1se)),
                           c(coef(lm1), rep(0, sum(omegas[2:3]))),
                           c(coef(lm2), rep(0, omegas[3])), 
                           coef(lm3)
),
source = factor(rep(c("True values", 
                      "Lasso", 
                      "Ridge regression", 
                      "Hierarchical ridge", 
                      "Linear model, \n main effects",
                      "Linear model, \n + second order interactions", 
                      "Linear model, \n + third order interactions"), 
                    each = (sum(omegas)+1)),
                levels= c("True values", "Lasso", "Ridge regression",
                          "Hierarchical ridge",
                          "Linear model, \n main effects",
                          "Linear model, \n + second order interactions",
                          "Linear model, \n + third order interactions") ),
index = c(rep(1:(sum(omegas) + 1), 7))
)

ylims <- max(abs(ggmat$values))*c(-1,1)
xlims <- c(1, sum(omegas)+1)

g1 <- ggplot(ggmat[1:(4*(sum(omegas) +1)),], aes(x = index, y = values)) +
  facet_wrap(~source, ncol = 3) +
  geom_hline(yintercept = 0 , color = "red", linetype = 3) +
  geom_point(size = .9) +
  geom_vline(xintercept = cumsum(omegas)+1, linetype = 2) +
  theme_bw() +
  theme(strip.text.x = element_text(size = 12),
        strip.background = element_blank(),
        legend.position="none",
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.title.y=element_blank(),
        plot.margin = unit(c(1,1,0,1), "lines")) +
  xlim(xlims) +
  ylim(ylims)

g1 <- ggplot_gtable(ggplot_build(g1))
g1$layout$clip[g1$layout$name %in% c("panel-1-1", "panel-2-1","panel-3-1",
                                     "panel-1-2", "panel-2-2","panel-3-2")] <- "off"

g2 <- ggplot(ggmat[(4*(sum(omegas)+1)+1):nrow(ggmat),], aes(x = index, y = values)) +
  facet_wrap(~source) +
  geom_hline(yintercept = 0 , color = "red", linetype = 3) +
  geom_point(size = .9) +
  theme_bw() +
  theme(strip.text.x = element_text(size = 12),
        strip.background = element_blank(),
        legend.position="none",
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.title.y=element_blank(),
        plot.margin = unit(c(0,1,2,1), "lines")) +
  xlim(xlims) +
  ylim(ylims) +
  # horizontal bin lines
  annotation_custom(grob = linesGrob(), xmin = 1, xmax = cumsum(omegas)[1],
                    ymin = ylims[1]-.5, ymax = ylims[1]-.5) +
  annotation_custom(grob = linesGrob(), xmin = cumsum(omegas)[1]+1,
                    xmax = cumsum(omegas)[2], ymin = ylims[1]-.5,
                    ymax = ylims[1]-.5) +
  annotation_custom(grob = linesGrob(), xmin = cumsum(omegas)[2]+1,
                    xmax = cumsum(omegas)[3], ymin = ylims[1]-.5,
                    ymax = ylims[1]-.5) +
  # vertical bin lines
  annotation_custom(grob = linesGrob(), xmin = 1, xmax = 1,
                    ymin = ylims[1]-.5, ymax = ylims[1]-.2) +
  annotation_custom(grob = linesGrob(), xmin = cumsum(omegas)[1],
                    xmax = cumsum(omegas)[1], ymin = ylims[1]-.5,
                    ymax = ylims[1]-.2) +
  annotation_custom(grob = linesGrob(), xmin = cumsum(omegas)[1]+1,
                    xmax = cumsum(omegas)[1]+1, ymin = ylims[1]-.5,
                    ymax = ylims[1]-.2) +
  annotation_custom(grob = linesGrob(), xmin = cumsum(omegas)[2],
                    xmax = cumsum(omegas)[2], ymin = ylims[1]-.5,
                    ymax = ylims[1]-.2) +
  annotation_custom(grob = linesGrob(), xmin = cumsum(omegas)[2]+1,
                    xmax = cumsum(omegas)[2]+1, ymin = ylims[1]-.5,
                    ymax = ylims[1]-.2) +
  annotation_custom(grob = linesGrob(), xmin = cumsum(omegas)[3],
                    xmax = cumsum(omegas)[3], ymin = ylims[1]-.5,
                    ymax = ylims[1]-.2) +
  # vertical bin ticks
  annotation_custom(grob = linesGrob(),
                    xmin = mean(c(1, omegas[1])),
                    xmax = mean(c(1, omegas[1])),
                    ymin = ylims[1]-.6,
                    ymax = ylims[1]-.35) +
  annotation_custom(grob = linesGrob(),
                    xmin = mean(c(cumsum(omegas)[1]+1, cumsum(omegas)[2])),
                    xmax = mean(c(cumsum(omegas)[1]+1, cumsum(omegas)[2])),
                    ymin = ylims[1]-.6,
                    ymax = ylims[1]-.35) +
  annotation_custom(grob = linesGrob(),
                    xmin = mean(c(cumsum(omegas)[2]+1, cumsum(omegas)[3])),
                    xmax = mean(c(cumsum(omegas)[2]+1, cumsum(omegas)[3])),
                    ymin = ylims[1]-.6,
                    ymax = ylims[1]-.35) +
  # bin text
  annotation_custom(textGrob("k = 1", gp = gpar(fontsize = 10)),
                    xmin = mean(c(1, omegas[1])),xmax = mean(c(1, omegas[1])),
                    ymin = ylims[1]-.8, ymax = ylims[1]-.8) +
  annotation_custom(textGrob("k = 2", gp = gpar(fontsize = 10)),
                    xmin = mean(c(cumsum(omegas)[1]+1, cumsum(omegas)[2])),
                    xmax = mean(c(cumsum(omegas)[1]+1, cumsum(omegas)[2])),
                    ymin = ylims[1]-.8, ymax = ylims[1]-.8) +
  annotation_custom(textGrob("k = 3", gp = gpar(fontsize = 10)),
                    xmin = mean(c(cumsum(omegas)[2]+1, cumsum(omegas)[3])),
                    xmax = mean(c(cumsum(omegas)[2]+1, cumsum(omegas)[3])),
                    ymin = ylims[1]-.8, ymax = ylims[1]-.8)


g2 <- ggplot_gtable(ggplot_build(g2))
g2$layout$clip[g2$layout$name %in% c("panel-1-1", "panel-2-1","panel-3-1")] <- "off"

g <- grid.arrange(g1, g2, heights = c(1.65, 1))
