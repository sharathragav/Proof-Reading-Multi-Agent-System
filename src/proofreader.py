import torch
def proofread_text(tokenizer, model, device, text, max_input=512, max_output=512):
    prompt = f"Edit: {text.strip()}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_output, do_sample=False, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()