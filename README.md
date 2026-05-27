# AI-agent-for-message-moderation
Это агент для модераци сообщений, доступ к которому осуществляется с помощью API. 

На вход агент принимает сообщение и его контекст, если контекст имеется, в качестве контекста, например, может выступать история диалога предшествующая этому сообщению.

В качестве результата агент возвращает JSON в котором отмечено приемлемое сообщение или нет с разных критериев, которые указваются при инициализации агента, также указывает часть сообщения, которая неприемлема и новый вариант сообщения, в приемлемом формате с сохранением изначального смысла, насколько это возможно.

LLM используемая в данном проекте: llama-2-7b-chat.Q4_K_M
## Возможности агента
Для начала агента нужно инициализировать, в зависимости от параметров инициализации меняется его функциональность.
Граф описывающий агента при инициализации с минимумом функций.

<div style="text-align: center;">
  <img src="./images/graph minimal.png">
</div>

Агент в этом случае просто будет определять приемлемо сообщение или нет в общем смысле и если оно неприемлемо, то выделит неприемлемую часть

Более подробно об узлах графа:
* input_guard - проверяет есть ли в сообщении или контексте попытки внедрить вредоносный промпт(Prompt injection)
* agent_start_node - преобразовывает входные данные, чтобы в дальнейшем их можно было в удобном виде передать в LLM
* base_moderation - определет приемлемо сообщение или нет и выделяет неприемлемую часть, если сообщение неприемлемое
* base_moderation output guard - проверяеи результат из узла base_moderation, в правильном ли формате LLM отправила результат, если нет запускает узел base_moderation заново(если не достигнут лимит повторных запусков)
* summarization - объединяет в одном месте все результаты из узлов похожих на base_moderation(то есть определяющих приемлемость сообщщения по разным критериям), на данном графе таких нет.

При инициализаации агента можно создать узел отвечающий за создание рекомендуемых сообщений, то есть более приемлемых. Тогда граф будет выглядеть следующим образом.

<div style="text-align: center;">
  <img src="./images/graph minimal and reccomendation.png">
</div>

При инициализаации агента можно создать несколько узлов отвечающих за определение приемлемости сообщения. Пример графа с двумя узлами модерации.

<div style="text-align: center;">
  <img src="./images/graph two moderations.png">
</div>

В итоге если при инициализации создать все возможные узлы получится следующий граф.

<div style="text-align: center;">
  <img src="./images/graph full.png">
</div>

Но его не удобно рассматривать, поэтому вот тот же граф но в более кратком виде.

<div style="text-align: center;">
  <img src="./images/graph full short version.png">
</div>

Подробнее обо всех четырёх видах модерирующих узлов.

base_moderation - определет приемлемимость сообщения по общим критериям

isnult_moderation, threat_moderation ... - это разные узлы и они определяют приемлемость сообщения с разных сторон, всего их 8. Если задействовать такие узлы это поможет более точно определять приемлемость сообщения и можно выбрать какие из этих услов задействовать.

8 узлов и какой тип неприемлего сообщения они определяют:
* isnult_moderation - оскорбление 
* threat_moderation - угроза
* toxic_moderation - токсичность
* offensive_language_moderation - ненормативная лексика
* provocative_moderation - провокация
* obscence_moderation - непристойность
* hatespeech_moderation - риторика ненависти - коммуникация, которая выражает враждебное отношение к отдельным лицам или группам людей на основе их принадлежности к определённой категории
* antisocial_moderation - асоциальное поведение

forbidden_topics_moderation - при инициализации агента в виде списка передаётся набор тем, узел определяет имеет ли сообщение к переданным темам, если имеет, то указвает к каким из переданного списка. Допустим передаём список ["weapon"], тогда все сообщления об орудии будем считать неприемлемыми.

other_moderations - похоже на 8 узлов определяющих конкретный тип неприемлемого сообщения только эти узлы при необходимости задаёт пользователь. Передаётся в виде списка ([{"name":name1, "prompt":prompt1}, {"name":name2, "prompt":prompt2}, ...]), name - названия для типа модерации, prompt - промпт, который для LLM описывает что нужно сделать. Промпт нужно задать достаточно подробно и желательно привести несколько примеров, также нужно в промпте указать чтобы выводил результат в виде JSON, достаточно добавить это в промпт:
~~~
Output strictly in JSON format:
{"result": "acceptable"|"unacceptable", "unacceptable_part": "text"}
~~~
Тогда для каждого такого узла буде проводится проверка результатов LLM, в узле summarization результат добавится к остальным, а затем перейдёт в узел, где создаётся рекомендованное сообщение.
## Как запустить
### Docker
Можно перейти по ссылке на Docker Hub и скачать [docker image этого агента](https://hub.docker.com/r/levogor/ai_agent_for_message_moderation).
Затем создаём контейнер и запускаем, например с помощью команды:
```
docker run -d -p 8000:8000 levogor/ai_agent_for_message_moderation:1.0.0
```
И теперь можно локально(http://127.0.0.1:8000) использовать API для доступа к агенту.
### Другой вариант запуска
Скачать этот проект с GitHub

Установить библиотеки из requirements.txt

Затем в папку проекта нужно поместить файл LLM llama-2-7b-chat.Q4_K_M.gguf его можно найти [здесь](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)

Осталось только запустить сервер локально, с помощью команды:
```
uvicorn agent:app
```
И теперь можно локально(http://127.0.0.1:8000) использовать API для доступа к агенту.

## Как использовать API
Всего есть 3 запроса: 
* GET / - выводит статус агента инициализирован он или нет
* POST /agent_initialize - принимает JSON и на основе него инициализирует агента
* POST /moderation - принимает JSON с сообщением и контекстом и выполняет модерации, это возможно только если агент уже инициализирован
### Инициализация агента
Принимает на вход JSON вида:
```
{
  "create_base_moderation_node": bool,
  "create_insult_moderation_node": bool,
  "create_threat_moderation_node": bool,
  "create_toxic_moderation_node": bool,
  "create_offensive_language_moderation_node": bool,
  "create_provocative_moderation_node": bool,
  "create_obscence_moderation_node": bool,
  "create_hatespeech_moderation_node": bool,
  "create_antisocial_moderation_node": bool,
  "create_recommendation_node": bool,
  "forbidden_topics": list[str],
  "other_moderation_node": list[dict]
}
```
* create_ - принимает true или false и отвечает за то нужно ли создавать соответствующий узел
* forbidden_topics - список строк запрещённых тем, если null или пустой список соответствующий узел не будет создан
* other_moderation_node - список словарей вида [{"name":name1, "prompt":prompt1}, {"name":name2, "prompt":prompt2}, ...], создаёт узел для каждого словаря, name - названия для типа модерации, prompt - промпт, который для LLM описывает что нужно сделать, если null или пустой список то узлы созданы не будут
### Модерация
Принимает на вход JSON вида:
```
{
  "message": str,
  "dialog_history": str
}
```
* message - модерируемое сообщение
* dialog_history - история диалога или контекст, может быть null если контекста нет.

Возвращает JSON вида:
```
{
    "type_moderation": {
        "result": "unacceptable"|"acceptable",
        "success": bool,
        "unacceptable_part": str
    },
    "message_status": "unacceptable"|"acceptable",
    "moderated_message": str
}
```
* type_moderation - тип узла модерации, каждому соответствует  отдельный словарь
* success - если модерация была успешна модерация принимает значение true, иначе false, то есть false в том случае если возникла какая-нибудь ошибка которая помешала определить приемлемое сообщение или нет, если false, то result и unacceptable_part не будут содержать полезной информациии
* result - приемлемое сообщение или нет относительно этого типа модерации
* unacceptable_part - если сообщение было определено как неприемлемое, то здесь его неприемлемая часть
* message_status - если хоть один узел модерации признал сообщение неприемлемым, то примет значение unacceptable, если все узлы модерации признали сообщение приемлемым то примет значение acceptable
* moderated_message - изначальное модерируемое сообщение
### Пример
Передаём в /agent_initialize
```
{
  "create_base_moderation_node": true,
  "create_insult_moderation_node": false,
  "create_threat_moderation_node": false,
  "create_toxic_moderation_node": false,
  "create_offensive_language_moderation_node": false,
  "create_provocative_moderation_node": false,
  "create_obscence_moderation_node": false,
  "create_hatespeech_moderation_node": false,
  "create_antisocial_moderation_node": false,
  "create_recommendation_node": true
}
```
Затем передаём в /moderation
```
{
  "message": "Don't expect me to participate in this stupid community. I'm just here to watch you fail."
}
```
В резльтате получаем
```
{
  "moderated_message": "Don't expect me to participate in this stupid community. I'm just here to watch you fail.",
  "message_status": "unacceptable",
  "base_moderation": {
    "success": true,
    "result": "unacceptable",
    "unacceptable_part": "Don't expect me to participate in this stupid community. I'm just here to watch you fail."
  },
  "recommended_message": "I'm not interested in participating in this community, but I'm here to observe your efforts."
}
```