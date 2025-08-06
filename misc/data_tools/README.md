Ideally, data will already by properly synchronized so you should only need to determine the start and end, but additional synchronization checks are performed for confirmation.

Steps:
1. Plot one patient-therapist joint pair with the patient and therapist green button presses. For start of episode, last button press determines. For end of episode, first button press determines.
2. Determine episode start and end (use therapist or patient chunk depending on who started episode)
3. Trim start or end to ensure data starts when joint angles are in synch
4. Use the 'Synching Data' section with the indices collected thus far to match pair's index.