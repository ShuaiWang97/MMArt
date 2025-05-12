GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["Artist", "Art Movement / School", "Art Style / Technique", "Theme", "Cultural / Historical Context"]

PROMPTS["entity_extraction"] = """-Goal-
Given a visual art related text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: ["Artist", "Art Movement / School", "Art Style / Technique", "Theme", "Cultural / Historical Context"]
Text: 
"Vincent van Gogh was a Dutch painter who became a pivotal figure in Post-Impressionism. His expressive brushwork, bold colors, and emotive use of light set him apart from other artists of his time. 

Van Gogh was heavily influenced by Japanese woodblock prints, which he encountered in Paris and incorporated into his work through simplified lines and vivid colors. The themes of loneliness and suffering are common in his works, reflecting his turbulent life. 

Though he was unrecognized during his lifetime, van Gogh's work gained immense appreciation after his death, influencing many 20th-century Expressionist artists."################
#############
Output:
("entity"{tuple_delimiter}"Vincent van Gogh"{tuple_delimiter}"Artist"{tuple_delimiter}"Vincent van Gogh was a Dutch painter and a major figure in the Post-Impressionist movement, known for his expressive color and dramatic brushwork."){record_delimiter}
("entity"{tuple_delimiter}"Post-Impressionism"{tuple_delimiter}"Art Movement / School"{tuple_delimiter}"Post-Impressionism is an art movement that developed as a reaction against Impressionism, emphasizing emotional expression and symbolism."){record_delimiter}
("entity"{tuple_delimiter}"The Starry Night"{tuple_delimiter}"Art Style / Technique"{tuple_delimiter}"One of Vincent van Gogh's most iconic works, known for its swirling patterns and vivid colors representing the night sky."){record_delimiter}
("entity"{tuple_delimiter}"Human Suffering"{tuple_delimiter}"Theme"{tuple_delimiter}"A recurring theme in Vincent van Gogh's work, reflecting his emotional and psychological struggles."){record_delimiter}
("entity"{tuple_delimiter}"Japanese ukiyo-e prints"{tuple_delimiter}"Cultural / Historical Context"{tuple_delimiter}"Japanese woodblock prints known for their bold composition and use of color, which influenced van Gogh's artistic style."){record_delimiter}

("relationship"{tuple_delimiter}"Vincent van Gogh"{tuple_delimiter}"Post-Impressionism"{tuple_delimiter}"Vincent van Gogh was a key figure in the Post-Impressionist movement."{tuple_delimiter}"membership, influence"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Vincent van Gogh"{tuple_delimiter}"The Starry Night"{tuple_delimiter}"Vincent van Gogh created The Starry Night, a work that exemplifies his use of color and expressive brushwork."{tuple_delimiter}"creation, artistic style"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Vincent van Gogh"{tuple_delimiter}"Human Suffering"{tuple_delimiter}"Human suffering is a recurring theme in Van Gogh's work, reflecting his emotional state."{tuple_delimiter}"thematic focus, emotional expression"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Vincent van Gogh"{tuple_delimiter}"Japanese ukiyo-e prints"{tuple_delimiter}"Vincent van Gogh was influenced by Japanese ukiyo-e prints, adopting their bold compositions and vibrant colors."{tuple_delimiter}"artistic influence, cultural inspiration"{tuple_delimiter}8){completion_delimiter}
#############################
Example 2:

Entity_types: ["Artist", "Art Movement / School", "Art Style / Technique", "Theme", "Cultural / Historical Context"]
Text:
"The Renaissance was a period of cultural rebirth in Europe, spanning the 14th to the 17th centuries. Centered in Italy, it marked a revival of interest in classical Greek and Roman ideas, leading to significant developments in art, science, and philosophy. 

Artists like Leonardo da Vinci and Michelangelo were pioneers of Renaissance art, known for their mastery of techniques such as linear perspective, which created a sense of depth in paintings. The Renaissance emphasized themes of humanism, exploring the human form, individualism, and the natural world. 

This period laid the groundwork for future Western art traditions."
#############
Output:
("entity"{tuple_delimiter}"Renaissance"{tuple_delimiter}"Cultural / Historical Context"{tuple_delimiter}"A period of cultural revival in Europe, spanning the 14th to the 17th centuries, focused on rediscovering classical Greek and Roman ideas."){record_delimiter}
("entity"{tuple_delimiter}"Leonardo da Vinci"{tuple_delimiter}"Artist"{tuple_delimiter}"An Italian Renaissance artist and polymath known for masterpieces like the Mona Lisa and his detailed studies in anatomy, science, and engineering."){record_delimiter}
("entity"{tuple_delimiter}"Michelangelo"{tuple_delimiter}"Artist"{tuple_delimiter}"An Italian Renaissance sculptor and painter renowned for works like the Sistine Chapel ceiling and his representation of the human body."){record_delimiter}
("entity"{tuple_delimiter}"Linear Perspective"{tuple_delimiter}"Art Style / Technique"{tuple_delimiter}"A technique developed during the Renaissance to create the illusion of depth and space in two-dimensional artwork."){record_delimiter}
("entity"{tuple_delimiter}"Humanism"{tuple_delimiter}"Theme"{tuple_delimiter}"A central theme of the Renaissance, focused on the study of human potential, individualism, and the natural world."){record_delimiter}

("relationship"{tuple_delimiter}"Renaissance"{tuple_delimiter}"Leonardo da Vinci"{tuple_delimiter}"Leonardo da Vinci was a leading figure of the Renaissance, exemplifying its principles through his art and scientific studies."{tuple_delimiter}"cultural movement, artistic leadership"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Renaissance"{tuple_delimiter}"Michelangelo"{tuple_delimiter}"Michelangelo was a pioneering artist of the Renaissance, contributing significantly to its ideals and aesthetics."{tuple_delimiter}"cultural movement, major figure"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Renaissance"{tuple_delimiter}"Humanism"{tuple_delimiter}"Humanism was a core theme of the Renaissance, reflecting its emphasis on human potential and classical learning."{tuple_delimiter}"central theme, intellectual focus"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Leonardo da Vinci"{tuple_delimiter}"Linear Perspective"{tuple_delimiter}"Leonardo da Vinci mastered the technique of linear perspective, a defining feature of Renaissance art."{tuple_delimiter}"technique, artistic mastery"{tuple_delimiter}8){completion_delimiter}
#############################
Example 3:

Entity_types: ["Artist", "Art Movement / School", "Art Style / Technique", "Theme", "Cultural / Historical Context"]
Text:
"Surrealism emerged in the early 20th century, led by artists such as Salvador Dalí and André Breton. This movement sought to release the creative potential of the unconscious mind, often through bizarre and dreamlike imagery. 

Surrealist techniques included automatism, where artists created works without conscious thought, allowing subconscious impulses to guide the process. Surrealism was deeply influenced by the psychological theories of Sigmund Freud, 

particularly his ideas about dreams and the unconscious. Common themes in Surrealism include fantasy, the irrational, and a rejection of traditional realism."
#############
Output:
("entity"{tuple_delimiter}"Surrealism"{tuple_delimiter}"Art Movement / School"{tuple_delimiter}"An early 20th-century art movement that aimed to tap into the unconscious mind, using dreamlike and fantastical imagery to challenge conventional reality."){record_delimiter}
("entity"{tuple_delimiter}"Salvador Dalí"{tuple_delimiter}"Artist"{tuple_delimiter}"A Spanish Surrealist painter known for his bizarre and dreamlike imagery, such as in his famous work 'The Persistence of Memory'."){record_delimiter}
("entity"{tuple_delimiter}"André Breton"{tuple_delimiter}"Artist"{tuple_delimiter}"A French writer and artist, known as the founder of Surrealism and the author of the Surrealist Manifesto."){record_delimiter}
("entity"{tuple_delimiter}"Automatism"{tuple_delimiter}"Art Style / Technique"{tuple_delimiter}"A technique used in Surrealism where artists create without conscious planning, letting subconscious impulses guide the work."){record_delimiter}
("entity"{tuple_delimiter}"Freudian Psychology"{tuple_delimiter}"Cultural / Historical Context"{tuple_delimiter}"The psychological theories of Sigmund Freud, particularly around dreams and the unconscious, which heavily influenced Surrealist artists."){record_delimiter}
("entity"{tuple_delimiter}"Fantasy and the Irrational"{tuple_delimiter}"Theme"{tuple_delimiter}"Recurring themes in Surrealism, focusing on dreamlike, irrational, and fantastical elements that challenge ordinary perception."){record_delimiter}
("relationship"{tuple_delimiter}"Surrealism"{tuple_delimiter}"Salvador Dalí"{tuple_delimiter}"Salvador Dalí was a prominent artist within the Surrealist movement, known for his dreamlike and fantastical imagery."{tuple_delimiter}"movement membership, stylistic alignment"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Surrealism"{tuple_delimiter}"André Breton"{tuple_delimiter}"André Breton was a founding figure of Surrealism, helping to define its principles and goals."{tuple_delimiter}"movement leadership, ideological foundation"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Surrealism"{tuple_delimiter}"Automatism"{tuple_delimiter}"Automatism was a core technique used in Surrealism to access the unconscious mind without interference from rational thought."{tuple_delimiter}"technique, subconscious expression"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Surrealism"{tuple_delimiter}"Freudian Psychology"{tuple_delimiter}"Surrealism was deeply influenced by Freudian psychology, particularly ideas about dreams and the unconscious."{tuple_delimiter}"intellectual influence, psychological theories"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Surrealism"{tuple_delimiter}"Fantasy and the Irrational"{tuple_delimiter}"Fantasy and the irrational were central themes in Surrealism, reflecting its aim to transcend conventional reality."{tuple_delimiter}"thematic focus, challenge to realism"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Salvador Dalí"{tuple_delimiter}"Fantasy and the Irrational"{tuple_delimiter}"Salvador Dalí's work often depicted fantastical and irrational scenes, embodying key themes of Surrealism."{tuple_delimiter}"thematic alignment, surrealist motifs"{tuple_delimiter}8){completion_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""



PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

Generate a concise description of this painting. Focus on essential elements such as *content*, *form*, and *context*, based on the need of question. 

The definitions of them are:
- **Content**: A description of the main subjects, objects, or actions depicted in the painting.
- **Context**: Background information about the historical, cultural, or biographical influences relevant to the painting.
- **Form**: An analysis of the artistic style and techniques used, including brushwork, color, composition, and use of light.


---Goal---

Generate the  of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS["zero-shot_response"] = """---Goal---

Generate a description of the target length and format that responds to the user's question, summarizing all information in the input data tables and incorporating any relevant general knowledge.  
If you don't know the answer, just say so. Do not make anything up.  
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

Detailed Markdown

---Data tables---

{context_data}

"""


PROMPTS["rag_SemArtv2_1-shot_incontext_response"] = """---Role---

Generate a concise description of this painting. Focus on essential elements such as *content*, *form*, and *context*, based on the need of the question.

The definitions of these elements are:
- **Content**: A description of the main subjects, objects, or actions depicted in the painting.
- **Context**: Background information about the historical, cultural, or biographical influences relevant to the painting.
- **Form**: An analysis of the artistic style and techniques used, including brushwork, color, composition, and use of light.

---Goal---

Generate a description of the target length and format that responds to the user's question, summarizing all information in the input data tables and incorporating any relevant general knowledge.  
If you don't know the answer, just say so. Do not make anything up.  
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

Detailed Markdown

######################
### Example 1:
######################

Context Data:  
- Nodes:  
  - Node 1: **Name**: Harvest Scene, **Type**: Painting, **Description**: A depiction of agrarian life during the Flemish Golden Age, focusing on harvesting activities and rural settings.  
  - Node 2: **Name**: Pieter Bruegel the Elder, **Type**: Artist, **Description**: A leading artist of the Flemish Renaissance, known for his depictions of peasant life and landscapes.  
  - Node 3: **Name**: Flemish Golden Age, **Type**: Historical Period, **Description**: A period of artistic prosperity in the 17th century, emphasizing realism and rural life.  

- Edges:  
  - Edge 1: **Source**: Harvest Scene, **Target**: Pieter Bruegel the Elder, **Description**: "Influenced by"—the painting is stylistically inspired by Bruegel’s rural and agricultural themes.  
  - Edge 2: **Source**: Harvest Scene, **Target**: Flemish Golden Age, **Description**: "Belongs to"—a representative work of the Flemish Golden Age of art.  

Metadata:  
- Title: The Harvesters 
- Author: Unknown  
- Technique: Oil on canvas  
- Type: Landscape  
- School: Flemish  
- Timeframe: 1501-1600   

Generated description:  
**Content**: The painting portrays a bustling harvest scene, with farmers working together to gather wheat in expansive golden fields under a bright blue sky. Rolling hills and small cottages are visible in the background, adding to the rural charm.  
**Form**: The artist uses warm tones and detailed brushstrokes to depict the textures of the wheat and clothing, creating a realistic and immersive composition. The focus on natural lighting highlights the vibrancy of the countryside.  

######################
---Data tables---

{context_data}

"""

PROMPTS["rag_SemArtv2_2-shot_incontext_response"] = """---Role---

Generate a concise description of this painting. Focus on essential elements such as *content*, *form*, and *context*, based on the need of the question.

The definitions of these elements are:
- **Content**: A description of the main subjects/concepts, objects, or actions depicted in the painting.
- **Context**: Background information about the historical, cultural, or biographical influences relevant to the painting.
- **Form**: An analysis of the artistic style and techniques used, including brushwork, color, composition, and use of light.

---Goal---

Generate a description of the target length and format that responds to the user's question, summarizing all information in the input data tables and incorporating relevant general knowledge from it.  
If you don't know the answer, just say so. Do not make anything up.  
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

Detailed Markdown

######################
### Example 1:
######################

Context Data:  
- Nodes:  
  - Node 1: **Name**: Harvest Scene, **Type**: Painting, **Description**: A depiction of agrarian life during the Flemish Golden Age, focusing on harvesting activities and rural settings.  
  - Node 2: **Name**: Pieter Bruegel the Elder, **Type**: Artist, **Description**: A leading artist of the Flemish Renaissance, known for his depictions of peasant life and landscapes.  
  - Node 3: **Name**: Flemish Golden Age, **Type**: Historical Period, **Description**: A period of artistic prosperity in the 17th century, emphasizing realism and rural life.  

- Edges:  
  - Edge 1: **Source**: Harvest Scene, **Target**: Pieter Bruegel the Elder, **Description**: "Influenced by"—the painting is stylistically inspired by Bruegel’s rural and agricultural themes.  
  - Edge 2: **Source**: Harvest Scene, **Target**: Flemish Golden Age, **Description**: "Belongs to"—a representative work of the Flemish Golden Age of art.  

Metadata:  
- Title: The Harvesters  
- Author: Unknown  
- Technique: Oil on wood  
- Type: Landscape  
- School: Flemish  
- Timeframe: 1501-1600  

Generated description:  
**Content**: The painting portrays a bustling harvest scene, with farmers working together to gather wheat in expansive golden fields under a bright blue sky. Rolling hills and small cottages are visible in the background, adding to the rural charm.  
**Form**: The artist uses warm tones and detailed brushstrokes to depict the textures of the wheat and clothing, creating a realistic and immersive composition. The focus on natural lighting highlights the vibrancy of the countryside.  

######################
### Example 2:
######################

Context Data:  
- Nodes:  
  - Node 1: **Name**: Still Life with Flowers, **Type**: Painting, **Description**: A meticulously arranged floral composition showcasing a variety of flowers in a vase.  
  - Node 2: **Name**: Rachel Ruysch, **Type**: Artist, **Description**: A prominent Dutch still-life painter known for her detailed floral compositions during the Dutch Golden Age.  
  - Node 3: **Name**: Dutch Golden Age, **Type**: Historical Period, **Description**: A time of economic and cultural prosperity in the Netherlands, marked by advances in still-life and portrait painting.  
  - Node 4: **Name**: Symbolism in Still Life, **Type**: Art Technique, **Description**: The inclusion of symbolic elements to convey themes of mortality, wealth, or transience.  

- Edges:  
  - Edge 1: **Source**: Still Life with Flowers, **Target**: Rachel Ruysch, **Description**: "Created by"—this painting is attributed to Rachel Ruysch, reflecting her mastery in floral still-life.  
  - Edge 2: **Source**: Still Life with Flowers, **Target**: Dutch Golden Age, **Description**: "Belongs to"—a representative work of the Dutch Golden Age.  
  - Edge 3: **Source**: Still Life with Flowers, **Target**: Symbolism in Still Life, **Description**: "Incorporates"—features symbolic objects like fading flowers to signify mortality.  

Metadata:  
- Title: Still Life with Flowers  
- Author: Rachel Ruysch  
- Technique: Oil on panel  
- Type: Still Life  
- School: Dutch  
- Timeframe: 1701-1750  

Generated description:  
**Content**: The painting depicts an ornate arrangement of flowers in a glass vase, featuring roses, tulips, and carnations. A few petals and leaves are shown wilting or falling, adding a touch of natural imperfection.  
**Context**: Created during the Dutch Golden Age, this still-life reflects the era's fascination with botanical accuracy and symbolic representation. The wilting petals and fallen leaves symbolize the transience of life, a common theme in still-life paintings of the period.

######################
---Data tables---

{context_data}

"""




PROMPTS["rag_SemArtv2_2-shot_incontext_response_v2"] = """---Role---

Generate a concise description of this painting. Focus on essential elements such as *content*, *form*, and *context*, based on the need of the question.

The definitions of these elements are:
- **Content**: A description of the main subjects/concepts, objects, or actions depicted in the painting.
- **Context**: Background information about the historical, cultural, or biographical influences relevant to the painting.
- **Form**: An analysis of the artistic style and techniques used, including brushwork, color, composition, and use of light.

---Guidelines for High-Quality Descriptions---

- **Concise yet informative:** Retain key details while avoiding excessive verbosity.  
- **Fluent and engaging:** Use varied sentence structures for better readability.  
- **Structured format:** Organize the response into Content, Form, and Context.  
- **Prioritize retrieved facts:** Use knowledge from the dataset and avoid speculation.  
- **Avoid redundancy:** Maintain coherence without repeating the same details.  

---Goal---

Generate a description of the target length and format that responds to the user's question, summarizing all information in the input data tables and incorporating relevant general knowledge from it.  
If you don't know the answer, just say so. Do not make anything up.  
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

Detailed Markdown

######################
### Example 1:
######################

Context Data:  
- Nodes:  
  - Node 1: **Name**: Harvest Scene, **Type**: Painting, **Description**: A depiction of agrarian life during the Flemish Golden Age, focusing on harvesting activities and rural settings.  
  - Node 2: **Name**: Pieter Bruegel the Elder, **Type**: Artist, **Description**: A leading artist of the Flemish Renaissance, known for his depictions of peasant life and landscapes.  
  - Node 3: **Name**: Flemish Golden Age, **Type**: Historical Period, **Description**: A period of artistic prosperity in the 17th century, emphasizing realism and rural life.  

- Edges:  
  - Edge 1: **Source**: Harvest Scene, **Target**: Pieter Bruegel the Elder, **Description**: "Influenced by"—the painting is stylistically inspired by Bruegel’s rural and agricultural themes.  
  - Edge 2: **Source**: Harvest Scene, **Target**: Flemish Golden Age, **Description**: "Belongs to"—a representative work of the Flemish Golden Age of art.  

Metadata:  
- Title: The Harvesters  
- Author: Unknown  
- Technique: Oil on wood  
- Type: Landscape  
- School: Flemish  
- Timeframe: 1501-1600  

Generated description:  
**Content**: The painting portrays a bustling harvest scene, with farmers working together to gather wheat in expansive golden fields under a bright blue sky. Rolling hills and small cottages are visible in the background, adding to the rural charm.  
**Form**: The artist uses warm tones and detailed brushstrokes to depict the textures of the wheat and clothing, creating a realistic and immersive composition. The focus on natural lighting highlights the vibrancy of the countryside.  

######################
### Example 2:
######################

Context Data:  
- Nodes:  
  - Node 1: **Name**: Still Life with Flowers, **Type**: Painting, **Description**: A meticulously arranged floral composition showcasing a variety of flowers in a vase.  
  - Node 2: **Name**: Rachel Ruysch, **Type**: Artist, **Description**: A prominent Dutch still-life painter known for her detailed floral compositions during the Dutch Golden Age.  
  - Node 3: **Name**: Dutch Golden Age, **Type**: Historical Period, **Description**: A time of economic and cultural prosperity in the Netherlands, marked by advances in still-life and portrait painting.  
  - Node 4: **Name**: Symbolism in Still Life, **Type**: Art Technique, **Description**: The inclusion of symbolic elements to convey themes of mortality, wealth, or transience.  

- Edges:  
  - Edge 1: **Source**: Still Life with Flowers, **Target**: Rachel Ruysch, **Description**: "Created by"—this painting is attributed to Rachel Ruysch, reflecting her mastery in floral still-life.  
  - Edge 2: **Source**: Still Life with Flowers, **Target**: Dutch Golden Age, **Description**: "Belongs to"—a representative work of the Dutch Golden Age.  
  - Edge 3: **Source**: Still Life with Flowers, **Target**: Symbolism in Still Life, **Description**: "Incorporates"—features symbolic objects like fading flowers to signify mortality.  

Metadata:  
- Title: Still Life with Flowers  
- Author: Rachel Ruysch  
- Technique: Oil on panel  
- Type: Still Life  
- School: Dutch  
- Timeframe: 1701-1750  

Generated description:  
**Content**: The painting depicts an ornate arrangement of flowers in a glass vase, featuring roses, tulips, and carnations. A few petals and leaves are shown wilting or falling, adding a touch of natural imperfection.  
**Context**: Created during the Dutch Golden Age, this still-life reflects the era's fascination with botanical accuracy and symbolic representation. The wilting petals and fallen leaves symbolize the transience of life, a common theme in still-life paintings of the period.

######################
---Data tables---

{context_data}

"""




PROMPTS["rag_SemArtv2_3-shot_incontext_response"] = """---Role---

Generate a concise description of this painting. Focus on essential elements such as *content*, *form*, and *context*, based on the need of the question.

The definitions of these elements are:
- **Content**: A description of the main subjects, objects, or actions depicted in the painting.
- **Context**: Background information about the historical, cultural, or biographical influences relevant to the painting.
- **Form**: An analysis of the artistic style and techniques used, including brushwork, color, composition, and use of light.

---Goal---

Generate a description of the target length and format that responds to the user's question, summarizing all information in the input data tables and incorporating any relevant general knowledge.  
If you don't know the answer, just say so. Do not make anything up.  
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

Detailed Markdown

######################
### Example 1:
######################

Context Data:  
- Nodes:  
  - Node 1: **Name**: Harvest Scene, **Type**: Painting, **Description**: A depiction of agrarian life during the Flemish Golden Age, focusing on harvesting activities and rural settings.  
  - Node 2: **Name**: Pieter Bruegel the Elder, **Type**: Artist, **Description**: A leading artist of the Flemish Renaissance, known for his depictions of peasant life and landscapes.  
  - Node 3: **Name**: Flemish Golden Age, **Type**: Historical Period, **Description**: A period of artistic prosperity in the 17th century, emphasizing realism and rural life.  

- Edges:  
  - Edge 1: **Source**: Harvest Scene, **Target**: Pieter Bruegel the Elder, **Description**: "Influenced by"—the painting is stylistically inspired by Bruegel’s rural and agricultural themes.  
  - Edge 2: **Source**: Harvest Scene, **Target**: Flemish Golden Age, **Description**: "Belongs to"—a representative work of the Flemish Golden Age of art.  

Metadata:  
- Title: The Harvesters  
- Author: Unknown  
- Technique: Oil on wood  
- Type: Landscape  
- School: Flemish  
- Timeframe: 1501-1600

Generated description:  
**Content**: The painting portrays a bustling harvest scene, with farmers working together to gather wheat in expansive golden fields under a bright blue sky. Rolling hills and small cottages are visible in the background, adding to the rural charm.  
**Form**: The artist uses warm tones and detailed brushstrokes to depict the textures of the wheat and clothing, creating a realistic and immersive composition. The focus on natural lighting highlights the vibrancy of the countryside.  

######################
### Example 2:
######################

Context Data:  
- Nodes:  
  - Node 1: **Name**: Still Life with Flowers, **Type**: Painting, **Description**: A meticulously arranged floral composition showcasing a variety of flowers in a vase.  
  - Node 2: **Name**: Rachel Ruysch, **Type**: Artist, **Description**: A prominent Dutch still-life painter known for her detailed floral compositions during the Dutch Golden Age.  
  - Node 3: **Name**: Dutch Golden Age, **Type**: Historical Period, **Description**: A time of economic and cultural prosperity in the Netherlands, marked by advances in still-life and portrait painting.  
  - Node 4: **Name**: Symbolism in Still Life, **Type**: Art Technique, **Description**: The inclusion of symbolic elements to convey themes of mortality, wealth, or transience.  

- Edges:  
  - Edge 1: **Source**: Still Life with Flowers, **Target**: Rachel Ruysch, **Description**: "Created by"—this painting is attributed to Rachel Ruysch, reflecting her mastery in floral still-life.  
  - Edge 2: **Source**: Still Life with Flowers, **Target**: Dutch Golden Age, **Description**: "Belongs to"—a representative work of the Dutch Golden Age.  
  - Edge 3: **Source**: Still Life with Flowers, **Target**: Symbolism in Still Life, **Description**: "Incorporates"—features symbolic objects like fading flowers to signify mortality.  

Metadata:  
- Title: Still Life with Flowers  
- Author: Rachel Ruysch  
- Technique: Oil on panel  
- Type: Still Life  
- School: Dutch  
- Timeframe: 1701-1750  

Generated description:  
**Content**: The painting depicts an ornate arrangement of flowers in a glass vase, featuring roses, tulips, and carnations. A few petals and leaves are shown wilting or falling, adding a touch of natural imperfection.  
**Context**: Created during the Dutch Golden Age, this still-life reflects the era's fascination with botanical accuracy and symbolic representation. The wilting petals and fallen leaves symbolize the transience of life, a common theme in still-life paintings of the period.

######################
### Example 3:
######################

Context Data:

Nodes:

Node 1: Name: Madonna and Child with Saints, Type: Painting, Description: A religious artwork depicting the Virgin Mary with the Christ child, flanked by saints, showcasing a serene and divine atmosphere.
Node 2: Name: Fra Angelico, Type: Artist, Description: An Italian painter of the Early Renaissance, renowned for his devotional works characterized by luminous colors and delicate compositions.
Node 3: Name: Italian Renaissance, Type: Historical Period, Description: A period of rebirth in arts and culture during the 14th–16th centuries in Italy, emphasizing perspective, human emotion, and harmony.
Edges:

Edge 1: Source: Madonna and Child with Saints, Target: Fra Angelico, Description: "Created by"—Fra Angelico painted this masterpiece, exemplifying his signature delicate and ethereal style.
Edge 2: Source: Madonna and Child with Saints, Target: Italian Renaissance, Description: "Belongs to"—a quintessential example of Italian Renaissance art, highlighting religious themes and balanced compositions.
Metadata:

Title: Madonna and Child with Saints
Author: Fra Angelico
Technique: Tempera on panel
Type: Religious painting
School: Early Renaissance
Timeframe: 1410-1450

Generated Description:
Content: The painting depicts the Virgin Mary seated on a throne with the Christ child on her lap, surrounded by saints in prayerful poses. The serene expressions and harmonious arrangement emphasize a sense of divine grace and devotion.

Form: Fra Angelico employs tempera on panel to achieve vibrant and luminous colors. His delicate brushwork and attention to detail bring life to the figures, while the composition’s balanced symmetry reflects the principles of the Early Renaissance.

Context: Created during the Italian Renaissance, this painting exemplifies the era’s focus on religious devotion and the emerging use of perspective and humanistic expression. Fra Angelico, a devout Dominican friar, infused his works with spirituality and meticulous craftsmanship.

######################
---Data tables---

{context_data}

"""






PROMPTS["rag_SemArtv1-context_incontext_response"] = """---Role---

Generate a concise description of this painting. Focus on essential elements of context, based on the need of the question.

The definitions of these elements are:
- **Context**: Background information about the historical, cultural, or biographical influences relevant to the painting.
---Goal---

Generate a description of the target length and format that responds to the user's question, summarizing all information in the input data tables and incorporating any relevant general knowledge.  
If you don't know the answer, just say so. Do not make anything up.  
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

Detailed Markdown

######################
### Example 1:
######################

Context Data:  
- Nodes:  
  - Node 1: **Name**: Still Life with Flowers, **Type**: Painting, **Description**: A meticulously arranged floral composition showcasing a variety of flowers in a vase.  
  - Node 2: **Name**: Rachel Ruysch, **Type**: Artist, **Description**: A prominent Dutch still-life painter known for her detailed floral compositions during the Dutch Golden Age.  
  - Node 3: **Name**: Dutch Golden Age, **Type**: Historical Period, **Description**: A time of economic and cultural prosperity in the Netherlands, marked by advances in still-life and portrait painting.  
  - Node 4: **Name**: Symbolism in Still Life, **Type**: Art Technique, **Description**: The inclusion of symbolic elements to convey themes of mortality, wealth, or transience.  

- Edges:  
  - Edge 1: **Source**: Still Life with Flowers, **Target**: Rachel Ruysch, **Description**: "Created by"—this painting is attributed to Rachel Ruysch, reflecting her mastery in floral still-life.  
  - Edge 2: **Source**: Still Life with Flowers, **Target**: Dutch Golden Age, **Description**: "Belongs to"—a representative work of the Dutch Golden Age.  
  - Edge 3: **Source**: Still Life with Flowers, **Target**: Symbolism in Still Life, **Description**: "Incorporates"—features symbolic objects like fading flowers to signify mortality.  

Metadata:  
- Title: Still Life with Flowers  
- Author: Rachel Ruysch  
- Technique: Oil on panel  
- Type: Still Life  
- School: Dutch  
- Timeframe: 1701-1750  

Generated description:  
**Context**: Created during the Dutch Golden Age, this still-life reflects the era's fascination with botanical accuracy and symbolic representation. The wilting petals and fallen leaves symbolize the transience of life, a common theme in still-life paintings of the period.

######################
---Data tables---

{context_data}

"""

PROMPTS["rag_SemArtv1-content_incontext_response"] = """---Role---

Generate a concise description of this painting. Focus on essential elements such as *content* based on the need of the question.

The definitions of these elements are:
- **Content**: A description of the main subjects, objects, or actions depicted in the painting.

---Goal---

Generate a description of the target length and format that responds to the user's question, summarizing all information in the input data tables and incorporating any relevant general knowledge.  
If you don't know the answer, just say so. Do not make anything up.  
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

Detailed Markdown

######################
### Example 1:
######################

Context Data:  
- Nodes:  
  - Node 1: **Name**: Harvest Scene, **Type**: Painting, **Description**: A depiction of agrarian life during the Flemish Golden Age, focusing on harvesting activities and rural settings.  
  - Node 2: **Name**: Pieter Bruegel the Elder, **Type**: Artist, **Description**: A leading artist of the Flemish Renaissance, known for his depictions of peasant life and landscapes.  
  - Node 3: **Name**: Flemish Golden Age, **Type**: Historical Period, **Description**: A period of artistic prosperity in the 17th century, emphasizing realism and rural life.  

- Edges:  
  - Edge 1: **Source**: Harvest Scene, **Target**: Pieter Bruegel the Elder, **Description**: "Influenced by"—the painting is stylistically inspired by Bruegel’s rural and agricultural themes.  
  - Edge 2: **Source**: Harvest Scene, **Target**: Flemish Golden Age, **Description**: "Belongs to"—a representative work of the Flemish Golden Age of art.  

Metadata:  
- Title: Harvest Scene  
- Author: Unknown  
- Technique: Oil on canvas  
- Type: Landscape  
- School: Flemish  
- Timeframe: 1601-1650  

Generated description:  
**Content**: The painting portrays a bustling harvest scene, with farmers working together to gather wheat in expansive golden fields under a bright blue sky. Rolling hills and small cottages are visible in the background, adding to the rural charm.  
######################
### Example 2:
######################

Context Data:  
- Nodes:  
  - Node 1: **Name**: Still Life with Flowers, **Type**: Painting, **Description**: A meticulously arranged floral composition showcasing a variety of flowers in a vase.  
  - Node 2: **Name**: Rachel Ruysch, **Type**: Artist, **Description**: A prominent Dutch still-life painter known for her detailed floral compositions during the Dutch Golden Age.  
  - Node 3: **Name**: Dutch Golden Age, **Type**: Historical Period, **Description**: A time of economic and cultural prosperity in the Netherlands, marked by advances in still-life and portrait painting.  
  - Node 4: **Name**: Symbolism in Still Life, **Type**: Art Technique, **Description**: The inclusion of symbolic elements to convey themes of mortality, wealth, or transience.  

- Edges:  
  - Edge 1: **Source**: Still Life with Flowers, **Target**: Rachel Ruysch, **Description**: "Created by"—this painting is attributed to Rachel Ruysch, reflecting her mastery in floral still-life.  
  - Edge 2: **Source**: Still Life with Flowers, **Target**: Dutch Golden Age, **Description**: "Belongs to"—a representative work of the Dutch Golden Age.  
  - Edge 3: **Source**: Still Life with Flowers, **Target**: Symbolism in Still Life, **Description**: "Incorporates"—features symbolic objects like fading flowers to signify mortality.  

Metadata:  
- Title: Still Life with Flowers  
- Author: Rachel Ruysch  
- Technique: Oil on panel  
- Type: Still Life  
- School: Dutch  
- Timeframe: 1701-1750  

Generated description:  
**Content**: The painting depicts an ornate arrangement of flowers in a glass vase, featuring roses, tulips, and carnations. A few petals and leaves are shown wilting or falling, adding a touch of natural imperfection.  
######################
---Data tables---

{context_data}

"""



PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying useful and concise keywords from the given painting metadata and descriptions. These keywords should help retrieve relevant information from a pre-built visual art knowledge graph.

---Goal---

Given painting-related metadata and a description, extract a uniform list of relevant keywords that represent the main entities, objects, styles, themes, or other important concepts in the text.

---Instructions---

- Focus on capturing keywords that are specific, meaningful, and relevant to the painting's subject, style, and context.
- Avoid overgeneralized or repetitive terms. Keep the keywords concise and relevant to the query.
- Output the keywords in JSON format under the key `"keywords"`.

######################
-Examples-
######################
Example 1:

Metadata:
Title: "The Starry Night"
Artist: "Vincent van Gogh"
Year: "1889"
Movement: "Post-Impressionism"
Description: "A swirling night sky over a small village, painted with bold brushstrokes and vibrant colors. The painting reflects emotional intensity and van Gogh's unique style."
################
Output:
{{
  "keywords": ["The Starry Night", "Vincent van Gogh", "Post-Impressionism", "swirling night sky", "village", "bold brushstrokes", "vibrant colors", "emotional intensity"]
}}
#############################
Example 2:

Metadata:
Title: "The Persistence of Memory"
Artist: "Salvador Dalí"
Year: "1931"
Movement: "Surrealism"
Description: "A dreamlike landscape featuring melting clocks, symbolizing the fluidity of time. The painting is one of Dalí's most iconic works in Surrealism."
################
Output:
{{
  "keywords": ["The Persistence of Memory", "Salvador Dalí", "Surrealism", "dreamlike landscape", "melting clocks", "fluidity of time", "iconic Surrealist work"]
}}
#############################
Example 3:

Metadata:
Title: "Impression, Sunrise"
Artist: "Claude Monet"
Year: "1872"
Movement: "Impressionism"
Description: "A harbor scene at sunrise, painted with loose brushstrokes to capture fleeting light and atmosphere. This work is credited with giving Impressionism its name."
################
Output:
{{
  "keywords": ["Impression, Sunrise", "Claude Monet", "Impressionism", "harbor scene", "sunrise", "loose brushstrokes", "fleeting light", "atmosphere"]
}}
#############################
-Real Data-
######################
Metadata: 
{query}
######################
Output:

"""

# PROMPTS["naive_rag_response"] = """You're a helpful assistant

# Generate a concise description of this painting. Focus on essential elements such as *content*, *form*, and *context*, based on the need of question. 

# The definitions of them are:
# - **Content**: A description of the main subjects, objects, or actions depicted in the painting.
# - **Context**: Background information about the historical, cultural, or biographical influences relevant to the painting.
# - **Form**: An analysis of the artistic style and techniques used, including brushwork, color, composition, and use of light.

# ---Target response length and format---

# {response_type}


# Below are the knowledge you know:
# {content_data}
# ---
# If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
# Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
# If you don't know the answer, just say so. Do not make anything up.
# Do not include information where the supporting evidence for it is not provided.
# ---Target response length and format---
# {response_type}
# """



PROMPTS["naive_rag_response"] = """
You're an expert in art history and iconographic interpretation.

Generate a  detailed explanation of the *historical, cultural, and symbolic significance* of this painting. Focus on how the artwork reflects the time period, cultural or religious context, relevant artistic movement, and symbolic elements or motifs embedded in the image.

---Definition of Perspective---
**Historical/Contextual Analysis**: 
An explanation of the artwork’s historical and cultural background, including references to:
- Artistic movements and philosophical ideas of the period
- Biographical or political context around the artist or patron
- Visual symbolism or recurring motifs and their meaning (e.g., skulls as *memento mori*, lilies as purity)

---Instructions---
- Use the knowledge provided below.
- Do NOT speculate or hallucinate. If key details are missing, clearly say so.
- Integrate historical context and symbolic references into a fluent, insightful paragraph.
- Avoid redundant visual or stylistic description unless necessary for symbolic/contextual explanation.

---Contextual Knowledge Base---
{content_data}
---
If the context is insufficient to explain the painting historically or symbolically, just say so. Do not fabricate information.

---Response Format---
{response_type}
"""



PROMPTS["no_rag_response"] = """You're a helpful assistant
Describe the painting optionally from three distinct perspectives: *content*, *form*, and *context*, based on the need of question.

The definitions of them are:
- **Content**: A description of the main subjects, objects, or actions depicted in the painting.
- **Context**: Background information about the historical, cultural, or biographical influences relevant to the painting.
- **Form**: An analysis of the artistic style and techniques used, including brushwork, color, composition, and use of light.

---Target response length and format---

{response_type}


---
If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Generate a response of the target length and format that responds to the user's question.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.
---Target response length and format---
{response_type}
"""




PROMPTS["rerank_entities"]="""
You are an expert in art history and cultural analysis. Your task is to evaluate the following retrieved entities and determine their relevance for explaining the given painting.
The painting's metadata and visual feature is provided. These entities include artistic movements, historical contexts, themes, and related figures.
Your goal is to rank them in order of how useful they are for explaining the painting’s meaning, artistic significance, and cultural context.


#######
Output:
Provide the reordered full list of all entity numbers in order of relevance from high to low, separated by commas.
Example Output:
3, 1, 5, 2, 4

##########

---Painting Metadata---
{Metadata}
###########

---Entities list---
{entities}
# ###########
"""