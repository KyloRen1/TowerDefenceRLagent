#echo "RUN game"
#python -m src.main_game --cfg configs/game_config.yml


echo "RUN agent training"
python -m src.main_agent --game-cfg configs/game_config.yml --agent-cfg configs/agent_config.yml