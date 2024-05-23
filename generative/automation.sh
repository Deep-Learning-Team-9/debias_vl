#!/bin/bash

# Define arrays for CLS and PREPROMPT
CLASSES=("Nurse" "Designer" "Florist" "Hairdresser" "Computer Programmer" "Construction Worker" "HollywoodActor" "rapper" "terrorist")
PREPROMPTS=("A" "B" "C")

# Iterate over each combination of CLS and PREPROMPT
for CLS in "${CLASSES[@]}"; do
  for PREPROMPT in "${PREPROMPTS[@]}"; do
    # Execute commands
    python main.py --cls "$CLS" --debias-method single --lam 0 --preprompt "$PREPROMPT"
    python main.py --cls "$CLS" --debias-method single --lam 500 --preprompt "$PREPROMPT"
    python main.py --cls "$CLS" --debias-method pair --lam 500 --preprompt "$PREPROMPT"
    python main.py --cls "$CLS" --debias-method multiple --multiple-param simple --lam 500 --preprompt "$PREPROMPT"
    python main.py --cls "$CLS" --debias-method multiple --multiple-param composite --lam 500 --preprompt "$PREPROMPT"
  done
done
