library(ggplot2)
library(dplyr)
library(latex2exp)  # TeX function

C_h <- read.csv2("C_h_experiment.csv")

C_h$moves_made <- C_h$moves_made %>% as.numeric()
C_h$time <- C_h$time %>% as.numeric()
C_h$C_value <- C_h$C_value %>% as.numeric()

C_h %>% group_by(C_value) %>% summarise(a = mean(moves_made, na.rm = TRUE))

C_h %>% 
  ggplot(aes(x=C_value, y=moves_made, color=factor(C_value),
             fill=factor(C_value), alpha=0.8)) +
  geom_dotplot(binaxis = "y", binwidth=25, stackdir="center") +
  labs(title = "Liczba ruchów w zależności od parametru C",
       subtitle = "którą potrzebował algorytm MCTS do pokonania losowo poruszającego się przeciwnika",
       x = "Wartosć parametru C",
       y = "Liczba ruchów do zwycięstwa") +
  scale_x_continuous(breaks = c(1, sqrt(2), 2, 3.5, 5),
                     labels = c("1", parse(text = TeX("$sqrt(2)$")), "2", "3.5", "5")) +
  stat_summary(fun = "mean", geom = "crossbar", width = 0.2) +
  theme_bw() +
  guides(color = "none", fill = "none", alpha = "none")



C_h_G <- read.csv2("C_h_G_experiment.csv")

C_h_G$moves_made <- C_h_G$moves_made %>% as.numeric()
C_h_G$G_value <- C_h_G$G_value %>% as.numeric()
C_h_G$time <- C_h_G$time %>% as.numeric()

C_h_G %>% group_by(G_value) %>%
  summarise(a = mean(moves_made, na.rm = TRUE)) %>% 
  arrange(G_value)

C_h_G %>% 
  ggplot(aes(x=G_value, y=moves_made, color=factor(G_value),
             fill=factor(G_value), alpha=0.8)) +
  geom_dotplot(binaxis = "y", binwidth=10, stackdir="center") +
  labs(title = "Liczba ruchów w zależności od parametru G",
       subtitle = "którą potrzebował algorytm MCTS do pokonania losowo poruszającego się przeciwnika",
       x = "Wartosć parametru G",
       y = "Liczba ruchów do zwycięstwa") +
  scale_x_continuous(breaks = c(1.1, 2, 3.5, 5, 7, 10, 20)) +
  stat_summary(fun = "mean", geom = "crossbar", width = 0.5) +
  theme_bw() +
  guides(color = "none", fill = "none", alpha = "none")




C_h_tournament <- read.csv2("C_h_comparison_experiment.csv")
C_h_tournament$who_won %>% table

C_h_G_tournament <- read.csv2("C_h_G_comparison_experiment.csv")
C_h_G_tournament$who_won %>% table

tournament_1 <- read.csv2("1_tournament_UCT_with_both_heuristics_experiment.csv")
tournament_1$who_won %>% table
