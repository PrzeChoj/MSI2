# MSI2
Repository for project of Artificial Intelligence Methods 2 - the course in MiNI WUT

## Project 1 - Solving CVRP problem with Ant Colony algorithm
All documents (conspect, raport and presentation) are available in folder `project1/documents`.

### Reproduction of the results
1. Clone this repository
2. Open the folder `project1/src/AntColony`
3. Run the comend: `pip install .`
4. Open the folder `project1/src/scripts`
5. Run the commend `python solve_multiple_problem.py`

After the point 5, you should see somenthing like this:
![Ongoing_script_photo_](project1/script_ongoing.png)

The script `solve_multiple_problem.py` has some parameters. One can see them with `python solve_multiple_problem.py help`

The script was executed with `max_time=300` and `number_of_repetitions=11`, which results with approximately 16 and a half hours of computing (300 seconds * 11 number\_of\_repetitions * 6 datasets * 3 modifictions = 16.5 hours).


## Project 2 - Solving the Taifho game with MCTS algorithm

