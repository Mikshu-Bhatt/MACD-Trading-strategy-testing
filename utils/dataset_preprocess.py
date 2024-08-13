import torch

nodes = {
    "Home Page": 0, "Bike Models Overview":1, "Specific Bike Model Page":2, "Build & Customize":3, 
    "Compare Models":4, "Find a Dealer":5, "Request a Quote":6, "Check Financing Options":7, 
    "Schedule a Test Ride":8, "Customer Reviews":9, "Accessory Shop":10, "Service & Maintenance Info":11, 
    "User Account Login/Registration":12, "Contact Support":13, "Promotions & Offers":14, 
    "Blog/News/Updates":15, "Events & Community":16, "Bike Registration":17, "Bike Insurance Information":18, 
    "FAQ/Help Center":19,'End of Journey':-1}


def preprocess_function_original_sp(tokenizer, max_length=512):
    def preprocess_function(examples):
        desc = [f"{d}" for d in examples['node_feat']]
        desc_ids = tokenizer(desc,
                             add_special_tokens=True,
                             truncation=True,
                             max_length=max_length,
                             padding='max_length',
                             return_tensors='pt', )
        return desc_ids

    return preprocess_function


def preprocess_function_generator_sp(tokenizer, prediction_type,ignore_index=0, max_length=32):
    def preprocess_function(examples):
        prompts = [f"You are given two things, first is dictionary of possible nodes and second is chat history between the user and the chatbot. based on this data, determine the most likely {prediction_type} from aal possible nodes in the user's journey. The dictionary of possible nodes is {nodes} and the chat history is given as: {feat}" for feat in examples['node_feat']]

        completion = [f"{l}" for l in examples['label']]
        instruction = f"\n\n###\n\n"
        instruction = tokenizer.encode(instruction, add_special_tokens=False)

        model_inputs = tokenizer(prompts, add_special_tokens=False)
        labels = tokenizer(completion, add_special_tokens=False)

        batch_size = len(examples['node_ids'])

        for i in range(batch_size):
            # Add bos & eos token
            sample_input_ids = [tokenizer.bos_token_id] + model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]

            p_max_length = max_length - len(label_input_ids) - len(instruction)
            sample_input_ids = sample_input_ids[:p_max_length] + instruction

            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [ignore_index] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                    max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"][i]
            labels["input_ids"][i] = [ignore_index] * (max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i])

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_function



def preprocess_test_function_generator_sp(tokenizer, prediction_type, max_length=32):
    def preprocess_function(examples):
        prompts = [f"You are given two things, first is dictionary of possible nodes and  second is chat history between the user and the chatbot. based on this data, determine the most likely {prediction_type} from aal possible nodes in the user's journey. The dictionary of possible nodes is {nodes} and the chat history is given as: {feat}" for feat in examples['node_feat']]
        model_inputs = tokenizer(prompts, add_special_tokens=False)

        instruction = f"\n\n###\n\n"
        instruction = tokenizer.encode(instruction, add_special_tokens=False)

        batch_size = len(examples['node_ids'])

        for i in range(batch_size):
            sample_input_ids = [tokenizer.bos_token_id] + model_inputs["input_ids"][i][
                                                          :max_length - len(instruction) - 1] + instruction

            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                    max_length - len(sample_input_ids)
            ) + sample_input_ids

            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + [1] * len(
                sample_input_ids)
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i])

        model_inputs['text_label'] = [f'{l}' for l in examples['label']]
        return model_inputs

    return preprocess_function


