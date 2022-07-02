# MSI2
Repository for the projects of Artificial Intelligence Methods 2 - the course in MiNI WUT.

Feel free to open the Issue in case of any problems with the source

## Project 1 - Solving CVRP problem with Ant Colony algorithm
All documents (Conspectus, Report and presentation) are available in the `project1/documents` folder.

### Reproduction of the results
1. Clone this repository
2. Open the folder `project1/src/AntColony`
3. Run the command: `pip install .`

After that, One has the `AntColony` package installed in one's Python. Now One can:

1. Run the Jupyter Notebooks from the folder `project1/src/JupyterNotebooks`
2. Run the script from `project1/src/scripts` with the commend ` Python solve_multiple_problem.py`

After initializing the script, one should see something like this:
![Ongoing_script_photo_](project1/script_ongoing.png)

The pipe (|) means that the found solution was better than the greedy solution. The dot (.) means that the found solution was worse than the greedy solution.

The script `solve_multiple_problem.py` has some parameters. One can inspect them with `python solve_multiple_problem.py help`

In the Report, the script was executed with `max_time=300` and `number_of_repetitions=11`, which results with approximately 16 and a half hours of computing (300 seconds * 11 number\_of\_repetitions * 6 datasets * 3 modifictions = 16.5 hours).

## Projekt 2 - Zastosowanie Upper Confidence Bound Applied To Trees do stworzenia sztucznej inteligencji grającej w grę Taifho dla dwóch graczy

### Jak zagrać z AI?
1. Pobierz to repozytorium
2. Przejdź w Terminalu do folderu `project2/src/Taifho`
3. W Terminalu wpisz komendę: `pip install .`
4. Przejdź w Terminalu do folderu `project2/src/Game_py`
5. W Terminalu wpisz komendę: `python main.py`
6. Wybierz opcję `[1] Start Game!` przyciskiem `1`
7. Wybierz algorytm MCTS oznaczony cyframi od 4 do 7. Cyfrą 1 oznaczono algorytm błądzenia losowego, a cyframi 2 i 3 podstawowe wersje algorytmu MCTS, o których pokazano w raporcie, że nie mogą działać satysfakcjonująco w grze Taifho.
8. Można dostosować pozostałe parametry algorytmu MCTS takie jak `C`, `G`, `steps` oraz czas "do namysłu" dla algorytmu MCTS, ale można zostawić je jako bazowe wybierając kilkakrotnie przycisk `Enter`
9. Wybierz którym kolorem chcesz grać. Zielony zaczyna (jak biały w szachach)

Pozycja startowa:

![starting_board_photo](project2/starting_board.png)

10. W każdym ruchu wybierz, którą bierką chcesz się ruszyć (można zmienić swój wybór) np. `D9` (można używać zarówno dużych jak i małych liter do oznaczenia pozycji)
11. Na planszy znakiem `*` pokazane zostały pola na które wybrana bierka może się ruszyć
12. Zaakceptuj wybór bierki przyciskiem `Enter`, bądź zmień bierkę przyciskiem `1`
13. Po wybraniu i zaakceptowaniu bierki, wybierz pozycję na którą chcesz ruszyć się bierką (jedna z tych oznaczonych gwiazdką)
14. Poczekaj, aż przeciwnik wykona swój ruch. Trwa to tyle, ile czasu zostało mu dane przed grą (domyślnie 10 sekund)
15. Znów wykonaj zwój ruch tak, jak od punktu 10
16. Jak gra się zakończy, to zdecyduj, czy chcesz zagrać jeszcze raz (być może z innymi parametrami) czy już chcesz zakończyć grę. W drugim przypadku wybierz opcję `[q] Quit`



