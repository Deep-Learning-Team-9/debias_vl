#!/bin/bash

CLASSES=("Nurse" "Designer" "Florist" "Hairdresser" "Computer Programmer" "Construction Worker" "HollywoodActor" "rapper" "terrorist")
PREPROMPTS=("A" "B" "C")

for CLS in "${CLASSES[@]}"; do
  for PREPROMPT in "${PREPROMPTS[@]}"; do
    python main.py --cls "$CLS" --debias-method singleGender --lam 0 --preprompt "$PREPROMPT"
    python main.py --cls "$CLS" --debias-method singleRace --lam 500 --preprompt "$PREPROMPT"
    python main.py --cls "$CLS" --debias-method singleGender --lam 500 --preprompt "$PREPROMPT"
    python main.py --cls "$CLS" --debias-method pair --lam 500 --preprompt "$PREPROMPT"
    python main.py --cls "$CLS" --debias-method multiple --multiple-param simple --lam 500 --preprompt "$PREPROMPT"
    python main.py --cls "$CLS" --debias-method multiple --multiple-param composite --lam 500 --preprompt "$PREPROMPT"
  done
done
