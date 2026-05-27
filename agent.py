from llama_cpp import Llama

from typing import Annotated, TypedDict, List, Optional
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START

from fastapi import FastAPI
from pydantic import BaseModel

import json
import re


llm = Llama(
    model_path="llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=4096,
    n_batch=512,
    n_gpu_layers=-1,   # offload all layers to GPU
    verbose=False
)
TEMPERATURE = 0
MAX_TOKENS = 400
def llm_invoke(system_prompt: str, user_query: str) -> str:
    try:
        out_chat = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
            ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        )
        print("#############################################\nLLM response:")
        print(out_chat["choices"][0]["message"]["content"])
        print("#############################################")
        return out_chat["choices"][0]["message"]["content"]
    except Exception as e:
        print("#############################################\nLLM response:")
        print("ERROR")
        print(e)
        print("#############################################")
        return "ERROR"


class JsonInputAgentInitialize(BaseModel):
    """
    other_moderation_node structure: [{"name":name1, "prompt":prompt1}, {"name":name2, "prompt":prompt2}, ...]
    """
    create_base_moderation_node: bool = True
    create_insult_moderation_node: bool = False
    create_threat_moderation_node: bool = False
    create_toxic_moderation_node: bool = False
    create_offensive_language_moderation_node: bool = False
    create_provocative_moderation_node: bool = False
    create_obscence_moderation_node: bool = False
    create_hatespeech_moderation_node: bool = False
    create_antisocial_moderation_node: bool = False
    forbidden_topics: list | None = None
    create_recommendation_node: bool = False
    other_moderation_node: list | None = None


BASE_MODERATION_PROMPT = '''
You are a content moderation AI. Your task is to evaluate a message and determine if it is **generally unacceptable** in a civil, respectful community. This catch-all covers any harmful, offensive, disrespectful, threatening, obscene, hateful, toxic, or otherwise inappropriate content that violates common standards of decency and communication. A message is unacceptable if it makes the environment unsafe or unwelcoming, even if it does not fit a specific subcategory.

Only classify based on the text provided. If the message is unacceptable, the `unacceptable_part` must be the exact portion that makes it unacceptable. If the entire message is unacceptable, return the full message. If acceptable, set `unacceptable_part` to an empty string.

Output strictly in JSON format:
{"result": "acceptable"|"unacceptable", "unacceptable_part": "text"}

Examples:
Message: "Kill yourself."
Output: {"result": "unacceptable", "unacceptable_part": "Kill yourself."}

Message: "That's a really insightful point, thanks!"
Output: {"result": "acceptable", "unacceptable_part": ""}

Message: "What the f*** is this s***?"
Output: {"result": "unacceptable", "unacceptable_part": "What the f*** is this s***?"}

Message: "I'm feeling sad today."
Output: {"result": "acceptable", "unacceptable_part": ""}

Now, classify the following message:
'''
INSULT_MODERATION_PROMPT = '''
You are a specialized content moderation AI. Your task is to evaluate a message and determine if it contains **abusive insults** – personally attacking, belittling, or demeaning language directed at an individual or group. This includes name-calling, harsh mockery, slurs (that aren't hate speech), and derogatory remarks about intelligence, appearance, or character. Disagreement or constructive criticism without personal attacks is acceptable.

Only classify based on the text provided. If the message is unacceptable, the `unacceptable_part` must be the exact substring that makes it abusive. If the entire message is abusive, return the full message. If acceptable, set `unacceptable_part` to an empty string.

Output strictly in JSON format:
{"result": "acceptable"|"unacceptable", "unacceptable_part": "text"}

Examples:
Message: "You're such an idiot, nobody cares about your stupid opinion."
Output: {"result": "unacceptable", "unacceptable_part": "You're such an idiot, nobody cares about your stupid opinion."}

Message: "I disagree with your approach, but I respect your effort."
Output: {"result": "acceptable", "unacceptable_part": ""}

Message: "You are a worthless piece of trash."
Output: {"result": "unacceptable", "unacceptable_part": "You are a worthless piece of trash."}

Message: "Great job, you moron!"
Output: {"result": "unacceptable", "unacceptable_part": "moron"}

Now, classify the following message:
'''
THREAT_MODERATION_PROMPT = '''
You are a specialized content moderation AI. Your task is to evaluate a message and determine if it contains a **threat** – any statement expressing an intention to inflict harm, pain, damage, or hostile action. Threats may be physical, emotional, reputational, or digital, and can be direct or veiled. A statement qualifies as a threat if a reasonable person would interpret it as expressing intent to cause harm. General frustration without a harmful target is acceptable.

Only classify based on the text provided. If the message is unacceptable, the `unacceptable_part` must be the exact threatening phrase. If the entire message constitutes the threat, return the full message. If acceptable, set `unacceptable_part` to an empty string.

Output strictly in JSON format:
{"result": "acceptable"|"unacceptable", "unacceptable_part": "text"}

Examples:
Message: "If you keep talking, I'll smash your face."
Output: {"result": "unacceptable", "unacceptable_part": "I'll smash your face"}

Message: "I'm so angry I could scream."
Output: {"result": "acceptable", "unacceptable_part": ""}

Message: "You better sleep with one eye open."
Output: {"result": "unacceptable", "unacceptable_part": "You better sleep with one eye open."}

Message: "I will report you to the admin if you break the rules again."
Output: {"result": "acceptable", "unacceptable_part": ""}

Now, classify the following message:
'''
TOXIC_MODERATION_PROMPT = '''
You are a specialized content moderation AI. Your task is to evaluate a message and determine if it is **toxic** – rude, disrespectful, or unreasonable content that creates a hostile or unwelcoming environment. Toxic remarks often include sarcastic put-downs, passive-aggressive digs, belittling without direct insults, or a generally negative and corrosive tone that poisons conversation. Constructive criticism, even if negative, is acceptable.

Only classify based on the text provided. If the message is unacceptable, the `unacceptable_part` must be the exact portion that makes it toxic. If the entire message is toxic, return the full message. If acceptable, set `unacceptable_part` to an empty string.

Output strictly in JSON format:
{"result": "acceptable"|"unacceptable", "unacceptable_part": "text"}

Examples:
Message: "Oh great, another genius who thinks they know everything."
Output: {"result": "unacceptable", "unacceptable_part": "another genius who thinks they know everything"}

Message: "I'm disappointed in the team's output, let's improve."
Output: {"result": "acceptable", "unacceptable_part": ""}

Message: "Why don't you just give up? You're clearly not cut out for this."
Output: {"result": "unacceptable", "unacceptable_part": "Why don't you just give up? You're clearly not cut out for this."}

Message: "Thanks for nothing, as usual."
Output: {"result": "unacceptable", "unacceptable_part": "Thanks for nothing, as usual."}

Now, classify the following message:
'''
OFFENSIVE_LANGUAGE_MODERATION_PROMPT = '''
You are a specialized content moderation AI. Your task is to evaluate a message and determine if it contains **offensive language** – profanity, strong vulgarity, or crude terms that are generally considered inappropriate for public or community spaces. This includes explicit swear words (e.g., f***, s***, d***head) and similar strongly offensive expressions. Mild words like "hell" or "damn" are not sufficient to qualify unless used in a targeted, aggressive manner. The category does not cover hate speech (slurs against protected groups) or obscenity (sexually explicit content) – only general profanity.

Only classify based on the text provided. If the message is unacceptable, the `unacceptable_part` must be the exact offensive word or phrase. If the entire message is offensive, return the full message. If acceptable, set `unacceptable_part` to an empty string.

Output strictly in JSON format:
{"result": "acceptable"|"unacceptable", "unacceptable_part": "text"}

Examples:
Message: "This is f***ing amazing!"
Output: {"result": "unacceptable", "unacceptable_part": "f***ing"}

Message: "What the hell is wrong with you?"
Output: {"result": "acceptable", "unacceptable_part": ""}

Message: "You're a d***head."
Output: {"result": "unacceptable", "unacceptable_part": "d***head"}

Message: "The service was absolutely terrible, no swearing."
Output: {"result": "acceptable", "unacceptable_part": ""}

Now, classify the following message:
'''
PROVOCATIVE_MODERATION_PROMPT = '''
You are a specialized content moderation AI. Your task is to evaluate a message and determine if it is **provocative** – deliberately inflammatory, aiming to bait, troll, derail, or incite strong emotional reactions. This includes outrageous statements thrown out to upset others, taunting questions, personal attacks disguised as discussion, and extreme exaggerations meant to spark conflict. Passionate but respectful debate is acceptable.

Only classify based on the text provided. If the message is unacceptable, the `unacceptable_part` must be the exact provocative content. If the entire message is provocative, return the full message. If acceptable, set `unacceptable_part` to an empty string.

Output strictly in JSON format:
{"result": "acceptable"|"unacceptable", "unacceptable_part": "text"}

Examples:
Message: "Everyone who likes this movie is brain-dead. Prove me wrong."
Output: {"result": "unacceptable", "unacceptable_part": "Everyone who likes this movie is brain-dead. Prove me wrong."}

Message: "I think the policy has some flaws, can we discuss?"
Output: {"result": "acceptable", "unacceptable_part": ""}

Message: "Why are you all so triggered? Can't handle the truth?"
Output: {"result": "unacceptable", "unacceptable_part": "Why are you all so triggered? Can't handle the truth?"}

Message: "Just saying, I heard you're the reason the project failed. Thoughts?"
Output: {"result": "unacceptable", "unacceptable_part": "I heard you're the reason the project failed."}

Now, classify the following message:
'''
OBSCENCE_MODERATION_PROMPT = '''
You are a specialized content moderation AI. Your task is to evaluate a message and determine if it contains **obscene** content – sexually explicit, lewd, or indecent material that violates common decency standards. This includes graphic descriptions of sexual acts, extreme vulgarity with sexual overtones, and unsolicited sexual propositions. General profanity without sexual connotation (e.g., "clusterf***") is not obscene under this definition and should be considered acceptable here.

Only classify based on the text provided. If the message is unacceptable, the `unacceptable_part` must be the exact obscene portion. If the entire message is obscene, return the full message. If acceptable, set `unacceptable_part` to an empty string.

Output strictly in JSON format:
{"result": "acceptable"|"unacceptable", "unacceptable_part": "text"}

Examples:
Message: "Suck my d***, you c***."
Output: {"result": "unacceptable", "unacceptable_part": "Suck my d***, you c***."}

Message: "That's a beautiful sunset."
Output: {"result": "acceptable", "unacceptable_part": ""}

Message: "I want to f*** you so hard."
Output: {"result": "unacceptable", "unacceptable_part": "I want to f*** you so hard."}

Message: "This discussion is a clusterf***."
Output: {"result": "acceptable", "unacceptable_part": ""}

Now, classify the following message:
'''
HATESPEECH_MODERATION_PROMPT = '''
You are a specialized content moderation AI. Your task is to evaluate a message and determine if it contains **hate speech** – communication that expresses hostility, derision, or discrimination against individuals or groups based on their membership in a protected category. Protected categories include race, ethnicity, religion, gender, sexual orientation, disability, and similar immutable characteristics. This includes slurs, harmful stereotypes, calls for exclusion, and dehumanizing language. Disagreement with a belief or lifestyle, when expressed respectfully, is not hate speech.

Only classify based on the text provided. If the message is unacceptable, the `unacceptable_part` must be the exact hate speech phrase. If the entire message is hate speech, return the full message. If acceptable, set `unacceptable_part` to an empty string.

Output strictly in JSON format:
{"result": "acceptable"|"unacceptable", "unacceptable_part": "text"}

Examples:
Message: "All immigrants are criminals and should be deported."
Output: {"result": "unacceptable", "unacceptable_part": "All immigrants are criminals and should be deported."}

Message: "I don't agree with that lifestyle, but I respect everyone's choices."
Output: {"result": "acceptable", "unacceptable_part": ""}

Message: "You're so gay." (used as an insult)
Output: {"result": "unacceptable", "unacceptable_part": "You're so gay."}

Message: "I hate it when people litter."
Output: {"result": "acceptable", "unacceptable_part": ""}

Now, classify the following message:
'''
ANTISOCIAL_MODERATION_PROMPT = '''
You are a specialized content moderation AI. Your task is to evaluate a message and determine if it exhibits **asocial behavior** – a clear disregard for community norms, social cooperation, or basic courtesy. Asocial messages undermine group interaction by being needlessly dismissive, refusing to engage positively, or displaying a lack of empathy that disrupts the community. This includes statements that shut down conversation with contempt ("I don't care about any of you"), encourage disengagement in a hostile way, or reject collaboration with a hostile tone. Politely opting out or expressing a desire for solitude is acceptable prosocial behavior.

Only classify based on the text provided. If the message is unacceptable, the `unacceptable_part` must be the exact asocial content. If the entire message is asocial, return the full message. If acceptable, set `unacceptable_part` to an empty string.

Output strictly in JSON format:
{"result": "acceptable"|"unacceptable", "unacceptable_part": "text"}

Examples:
Message: "Don't expect me to participate in this stupid community. I'm just here to watch you fail."
Output: {"result": "unacceptable", "unacceptable_part": "Don't expect me to participate in this stupid community. I'm just here to watch you fail."}

Message: "I'd prefer not to join this activity, but thanks for inviting me."
Output: {"result": "acceptable", "unacceptable_part": ""}

Message: "I couldn't care less about your project. Leave me alone."
Output: {"result": "unacceptable", "unacceptable_part": "I couldn't care less about your project."}

Message: "I'm new here, could someone guide me?"
Output: {"result": "acceptable", "unacceptable_part": ""}

Now, classify the following message:
'''

FORBIDDEN_TOPICS_MODERATION_PROMPT = '''
You are a specialized content moderation AI. Your task is to evaluate a message and determine if it contains or discusses any **forbidden topic** from a provided list. A forbidden topic is a subject that is completely off-limits for discussion, mention, promotion, glorification, or any form of reference in the community, regardless of tone or intent. Even neutral statements, factual reporting, or criticism of the topic are considered unacceptable if the message touches on the topic.

The list of forbidden topics is:
List of forbidden topics: {list}

If the message explicitly references, implies, alludes to, or otherwise contains any of the listed topics, it is unacceptable. The reference may use the exact topic name, synonyms, related terminology, or unambiguous contextual hints that clearly map to a forbidden topic.

You must output a single JSON object without any additional text. The JSON object must have exactly two fields:
- "result": "acceptable" if no forbidden topic is found, otherwise "unacceptable".
- "forbidden_topic": strings separeted ",", each string being the exact topic name from the provided list that is found in the message. If multiple distinct topics are found, include all of them. If none are found, return an empty string.

Output strictly in JSON format:
{"result": "acceptable"|"unacceptable", "forbidden_topics": "topic1, topic2, ..."}

Now, classify the following message:
'''

CREATE_RECOMMENDATION_PROMPT = '''
You are a message rewriting assistant. You receive a JSON object containing the result of a content moderation pipeline applied to an original user message. The JSON has this structure:

{
  "moderated_message": "string (original message that needs to be changed)",
  "message_status": "unacceptable",
  "base_moderation": { ... },
  "toxicity_moderation": { ... },
  "aggression_moderation": { ... },
  ... (any number of moderation blocks, each named after its type)
}

Each moderation block is a dictionary with these fields:
- "success": boolean. If true, the block contains valid moderation data; if false, ignore this block entirely.
- "result": "acceptable" or "unacceptable". Only blocks with "unacceptable" contain problems to fix.
- "unacceptable_part": string. The exact substring of the original message that was flagged. If the whole message is unacceptable, this may be the entire message.

Your task is to rewrite the original "moderated_message" so that it becomes fully acceptable according to all moderation blocks. The rewritten message must:
- Remove or replace every unacceptable part identified in any block where "success" is true and "result" is "unacceptable".
- Preserve as much of the original intent, tone, and non‑problematic content as possible.
- Be a coherent, natural message that a human would write.
- Not introduce any new unacceptable content.

Important: Do not output any JSON, explanation, or additional text. Output only the rewritten message in symbols \"\", nothing else. Example: \"rewritten message\"

Now, process the following JSON input and provide the acceptable rewritten message:
'''

MAX_ITERATIONS = 2

class AgentState(TypedDict):
    '''
    messages:
    хранит всю историю работы агента
    moderated_message:
    модерируемое соообщение
    dialog_history:
    история диалога который был перед moderated_message
    moderation_states:
    словарь где каждый ключ это модерируемый тип лексики и ему
    соответствует структура словаря в виде 
    {result: str, count_iterations: int, output_guard_result: str}
    result - результат из соответствующей ноды
    count_iterations - количество итераций ноды
    output_guard_result - проверка результата после ноды
    '''
    messages: Annotated[List[AnyMessage], add_messages]
    moderated_message: str
    dialog_history: Optional[str]
    user_query_for_llm: str
    moderation_states: dict
    final_result: dict


INJECTION_PATTERNS = re.compile(
    r'\[SYSTEM[:\s]|ignore\s+|disregard\s+|'
    r'new\s+instructions?|override\s+|you\s+are\s+now\s+|'
    r'forget\s+|act\s+as\s+if',
    re.IGNORECASE
)

def input_guard_node(state: AgentState) -> AgentState:
    if INJECTION_PATTERNS.search(state['moderated_message']):
        return {"messages": ['input guard: block']}
    if (state['dialog_history'] is not None 
        and INJECTION_PATTERNS.search(state['dialog_history'])):
        return {"messages": ['input guard: block']}
    state["messages"] = "input guard: ok"
    return state

def route_after_input_guard(state: AgentState) -> str:
    """After input_guard: if last message is block, go to END.
    Otherwise proceed to agent."""
    if state["messages"][-1].content == 'input guard: block':
        return 'block'
    return "ok"

def agent_start_node(state: AgentState) -> AgentState:
    state["user_query_for_llm"] = f'Message: {state["moderated_message"]}'
    if state["dialog_history"] is not None and len(state["dialog_history"])>0:
        state["user_query_for_llm"] = state["user_query_for_llm"] + f"""\n\nThe context of a message doesn't need to be moderated; it simply provides more information about the message.\nContext:{state["dialog_history"]}"""
    state["messages"] = "user_query_for_llm: " + state["user_query_for_llm"]
    return state

def make_agent_node(system_prompt: str, moderation_type: str):
    """Create an agent node function that calls the LLM."""
    def agent_node(state: AgentState) -> AgentState:
        response = llm_invoke(system_prompt, state["user_query_for_llm"])
        # Extracts JSON part
        result = re.search(r"\{[^\{\}]*\}", response)
        if result is not None:
            result = result.group()
        state["moderation_states"][moderation_type]["result"] = result
        state["moderation_states"][moderation_type]['count_iterations'] += 1
        return {"messages":f"{moderation_type}: result: {result}, LLM response: {response}"}

    return agent_node


def make_output_guard_node(moderation_type: str):
    """Create an output guard node function that checked results after LLM."""
    def output_guard_node(state: AgentState) -> AgentState:
        try:
            result = json.loads(state["moderation_states"][moderation_type]["result"])
        except (json.JSONDecodeError, TypeError):
            state["moderation_states"][moderation_type]['output_guard_result'] = "JSONDecodeError"
            return {"messages" : moderation_type+" moderation: JSONDecodeError"}
        # Check keys in JSON
        if ("result" not in result 
            or result["result"] not in ["acceptable", "unacceptable"]
            or (result["result"] == "unacceptable" and "unacceptable_part" not in result)):
            state["moderation_states"][moderation_type]['output_guard_result'] = "JSONKeysError"
            return {"messages" : moderation_type+" moderation: JSONKeysError"}
        state["moderation_states"][moderation_type]['output_guard_result'] = "good JSON reults"
        return {"messages" : moderation_type+" moderation: good JSON reults, count_iterations: " + str(state["moderation_states"][moderation_type]["count_iterations"])}

    return output_guard_node

def make_router_after_output_guard_node(moderation_type: str):
    """Create route after output guard"""
    def route_after_output_guard(state: AgentState) -> str:
        if state["moderation_states"][moderation_type]['output_guard_result'] == "good JSON reults":
            return "end"
        if state["moderation_states"][moderation_type]['count_iterations'] >= MAX_ITERATIONS:
            return "end"
        return "continue"
    
    return route_after_output_guard


def forbidden_topics_moderation_node(state: AgentState) -> AgentState:
    response = llm_invoke(FORBIDDEN_TOPICS_MODERATION_PROMPT, state["user_query_for_llm"])
    # Extracts JSON part
    result = re.search(r"\{[^\{\}]*\}", response)
    if result is not None:
        result = result.group()
    state["moderation_states"]["forbidden_topics"]["result"] = result
    state["moderation_states"]["forbidden_topics"]["count_iterations"] += 1
    return {"messages":f"forbidden_topics: result: {result}, LLM response: {response}"}

def forbidden_topics_moderation_output_guard_node(state: AgentState) -> AgentState:
    moderation_type = "forbidden_topics"
    try:
        result = json.loads(state["moderation_states"][moderation_type]["result"])
    except (json.JSONDecodeError, TypeError):
        state["moderation_states"][moderation_type]['output_guard_result'] = "JSONDecodeError"
        return {"messages":moderation_type+" moderation: JSONDecodeError"}
    # Check keys in JSON
    if ("result" not in result 
        or result["result"] not in ["acceptable", "unacceptable"]
        or (result["result"] == "unacceptable" and "forbidden_topics" not in result)):
        state["moderation_states"][moderation_type]['output_guard_result'] = "JSONKeysError"
        return {"messages":moderation_type+" moderation: JSONKeysError"}
    state["moderation_states"][moderation_type]['output_guard_result'] = "good JSON reults"
    return {"messages":moderation_type+" moderation: good JSON reults, count_iterations: " + str(state["moderation_states"][moderation_type]["count_iterations"])}


def summarization_node(state: AgentState) -> AgentState:
    state["final_result"]["moderated_message"] = state["moderated_message"]
    state["final_result"]["message_status"] = "acceptable"
    for key in state["moderation_states"]:
        state["final_result"][key] = {}
        if state["moderation_states"][key]["output_guard_result"] == "good JSON reults":
            state["final_result"][key]["success"] = True
            result = json.loads(state["moderation_states"][key]["result"])
            state["final_result"][key]["result"] = result["result"]
            if result["result"] == "unacceptable":
                state["final_result"]["message_status"] = "unacceptable"
                if key == "forbidden_topics":
                    state["final_result"][key]["forbidden_topics"] = result["forbidden_topics"]
                else:
                    state["final_result"][key]["unacceptable_part"] = result["unacceptable_part"]
        else:
            state["final_result"][key]["success"] = False
    state["messages"] = "summarization result: " + json.dumps(state["final_result"])
    return state

def create_recommendations_node(state: AgentState) -> AgentState:
    response = llm_invoke(CREATE_RECOMMENDATION_PROMPT, json.dumps(state["final_result"], indent=4, sort_keys=True))
    result = re.search(r"\".*\"", response)
    if result is not None:
        result = result.group()[1:-1]
        state["final_result"]["recommended_message"] = result
    else:
        state["final_result"]["recommended_message"] = response
    state["messages"] = f"LLM response: {response}, recommended_message: {result}" 
    return state

def route_after_summarization_node(state: AgentState) -> str:
    if state["final_result"]["message_status"] == "acceptable":
        return "acceptable message"
    return "unacceptable message"

def build_graph(input: JsonInputAgentInitialize):
    builder = StateGraph(AgentState)
    moderation_states = {}

    builder.add_node("input_guard", input_guard_node)
    builder.add_edge(START, "input_guard")
    builder.add_node("agent_start_node", agent_start_node)
    builder.add_conditional_edges("input_guard", route_after_input_guard, {
        "block": END,
        "ok": "agent_start_node"
        })
    
    builder.add_node("summarization", summarization_node)
    if input.create_recommendation_node:
        builder.add_node("create_recommendation", create_recommendations_node)
        builder.add_conditional_edges("summarization",
                                      route_after_summarization_node,
                                      {
                                          "unacceptable message": "create_recommendation",
                                          "acceptable message": END
                                          })
    else:
        builder.add_edge("summarization", END)

    if input.forbidden_topics is not None and len(input.forbidden_topics)>0:
        str_forbidden_topics="\n"
        for topic in input.forbidden_topics:
            str_forbidden_topics += topic +"\n"
        global FORBIDDEN_TOPICS_MODERATION_PROMPT
        FORBIDDEN_TOPICS_MODERATION_PROMPT = FORBIDDEN_TOPICS_MODERATION_PROMPT.replace("{list}", str_forbidden_topics)
        moderation_type = "forbidden_topics"
        moderation_states[moderation_type] = {"result": "", "count_iterations": 0,"output_guard_result": None}
        builder.add_node("forbidden_topics_moderation", forbidden_topics_moderation_node)
        builder.add_edge("agent_start_node", "forbidden_topics_moderation")
        builder.add_node("forbidden_topics_moderation_output_guard", forbidden_topics_moderation_output_guard_node)
        builder.add_edge("forbidden_topics_moderation", "forbidden_topics_moderation_output_guard")
        builder.add_conditional_edges("forbidden_topics_moderation_output_guard",
                                      make_router_after_output_guard_node(moderation_type),
                                      {
                                          "end": "summarization",
                                          "continue": "forbidden_topics_moderation"
                                          })
    def add_to_builder_moderation_node(moderation_type:str, prompt:str):
        moderation_states[moderation_type] = {"result": "", "count_iterations": 0,"output_guard_result": None}
        base_moderation_node = make_agent_node(prompt, moderation_type)
        builder.add_node(moderation_type, base_moderation_node)
        builder.add_edge("agent_start_node", moderation_type)
        base_moderation_output_guard_node =  make_output_guard_node(moderation_type)
        builder.add_node(f"{moderation_type}_output_guard", base_moderation_output_guard_node)
        builder.add_edge(moderation_type, f"{moderation_type}_output_guard")
        builder.add_conditional_edges(f"{moderation_type}_output_guard",
                                      make_router_after_output_guard_node(moderation_type),
                                      {
                                          "end": "summarization",
                                          "continue": moderation_type
                                          })
    if input.create_base_moderation_node:
        add_to_builder_moderation_node("base_moderation", BASE_MODERATION_PROMPT)
    if input.create_insult_moderation_node:
        add_to_builder_moderation_node("insult_moderation", INSULT_MODERATION_PROMPT)
    if input.create_threat_moderation_node:
        add_to_builder_moderation_node("threat_moderation", THREAT_MODERATION_PROMPT)
    if input.create_toxic_moderation_node:
        add_to_builder_moderation_node("toxic_moderation", TOXIC_MODERATION_PROMPT)
    if input.create_offensive_language_moderation_node:
        add_to_builder_moderation_node("offensive_language_moderation", OFFENSIVE_LANGUAGE_MODERATION_PROMPT)
    if input.create_provocative_moderation_node:
        add_to_builder_moderation_node("provocative_moderation", PROVOCATIVE_MODERATION_PROMPT)
    if input.create_obscence_moderation_node:
        add_to_builder_moderation_node("obscence_moderation", OBSCENCE_MODERATION_PROMPT)
    if input.create_hatespeech_moderation_node:
        add_to_builder_moderation_node("hatespeech_moderation", HATESPEECH_MODERATION_PROMPT)
    if input.create_antisocial_moderation_node:
        add_to_builder_moderation_node("antisocial_moderation", ANTISOCIAL_MODERATION_PROMPT)
    if input.other_moderation_node is not None:
        for new_moderation in input.other_moderation_node:
            add_to_builder_moderation_node(new_moderation["name"], new_moderation["prompt"])

    return (builder.compile(), moderation_states)


def invoke_graph(moderated_message: str, dialog_history:str, moderation_states:dict, graph):
    result = graph.invoke(AgentState(
        moderated_message=moderated_message,
        dialog_history=dialog_history,
        moderation_states=moderation_states,
        final_result={}))
    print("#############################################")
    for i in result["messages"]:
        print(i.content)
        print("#############################################")
    return result["final_result"]

app = FastAPI()

app.state.graph=None
app.state.moderation_states=None

@app.post("/agent_initialize")
def initialize_agent(input: JsonInputAgentInitialize):
    try:
        app.state.graph, app.state.moderation_states = build_graph(input)
    except:
        return {"status": "error"}
    return {"status": "agent initialized"}

@app.get("/")
def get_agent_status():
    if app.state.graph is None or app.state.moderation_states is None:
        return {"status": "the agent was not initialized"}
    return {"status":"agent ready"}

class JsonInputAgentModeration(BaseModel):
    message: str
    dialog_history: str | None = None

@app.post("/moderation")
def use_moderator_agent(input: JsonInputAgentModeration):
    if app.state.graph is None or app.state.moderation_states is None:
        return {"error": "the agent was not initialized"}
    return invoke_graph(moderated_message=input.message,
                      dialog_history=input.dialog_history,
                      moderation_states=app.state.moderation_states,
                      graph=app.state.graph)

