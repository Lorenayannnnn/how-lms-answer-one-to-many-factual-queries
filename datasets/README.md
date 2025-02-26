# Datasets

Following is an example line from the jsonl dataset file:
```
{
"query": "List only the name of three cities from Switzerland: 1.",         # Input query
"query_input_ids": [3231, 865, 272, 1141, 302, 1712, 9245, 477, 22491, 28747, 28705, 28740, 28723],     # Input query token ids 
"target_answer_idx": 1,     # answer step under examination
"target_answer_name": "Zurich",         # string of current step's answer
"target_answer_tokens": [25571, 539],   # token ids of the current step's answer
"three_answers_label_list": ["Zurich", "Geneva", "Bern"],       # list of all three answers
"three_answers_token_list": [[25571, 539], [6242, 13237], [7562]],  # token ids of all three answers
"three_answers_start_end_idx": [[13, 15], [18, 20], [23, 24]],  # start and end index of all three answers
"subject": "Switzerland",   # subject of the query
"subject_token_list": [22491],  # token ids of the subject
"subject_start_end_idx": [8, 9]   # start and end index of the subject
"relation_start_end_idx": [6, 8]   # start and end index of the relation (e.g.: cities, songs, movies)
}
```