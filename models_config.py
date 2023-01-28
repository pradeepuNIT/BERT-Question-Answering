
MODELS_CONFIG = {
	"config_file": "SQUAD_DIR/models/config.json",
	"models": {
		"bert_base_uncased_squad1.1_0": {
			"bert_architecture": "bert_base_uncased",
			"version": 'v1.1',
			"model_number": "0"},

		"bert_base_uncased_squad2.0_0": {
			"bert_architecture": "bert_base_uncased",
			"version": 'v2.0',
			"model_number": "0"},

		"traydstream_ucp_trained_on_bert_base": {
			"bert_architecture": "bert_base_uncased",
			"version": 'traydstream_ucp600',
			"model_number": "0"},
		"traydstream_ucp_trained_on_squadv1": {
			"bert_architecture": "bert_base_uncased",
			"version": 'traydstream_ucp600',
			"model_number": "1"}
	}
}
