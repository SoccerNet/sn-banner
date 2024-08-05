Evaluation on sn-gamestate

Follow instructions [here](https://github.com/SoccerNet/sn-gamestate/tree/main?tab=readme-ov-file#manual-downloading-of-soccernet-gamestate) to manually dowlnoad the SoccerNet gamestate dataset.
Only the "test" and "valid" splits are needed for the evaluation.
"valid" is used to find the best hyperparameters and "test" is used to evaluate the models and filters.

Execute with the following command:
```bash
bash find_window_filter_length.sh
```