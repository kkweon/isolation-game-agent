#!/bin/bash
rm -rf submit/
mkdir submit/
cp game_agent.py submit/
cp documents/heuristic_analysis.pdf submit/
cp documents/research_review.pdf submit/
zip submit/project2_KKM.zip submit/*
cd submit
