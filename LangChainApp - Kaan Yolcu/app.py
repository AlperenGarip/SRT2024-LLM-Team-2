from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
import os
import random

# Load environment variables from .env file
load_dotenv()

# Example long text to summarize
text_to_summarize = """
Rusya Federasyonu Devlet Marşı (Rusça: Государственный гимн Российской Федерации, Gosudarstvenniy gimn Rossiyskoy federatysii), Rusya'nın resmî devlet marşı. Bestesinde, Aleksandr Aleksandrov'a ait Jit stalo luçşe (Türkçe: Hayat daha iyi hâle geldi) ve Bolşevik Parti Marşı'ndan uyarlanan Sovyetler Birliği Marşı'nın müziği, güftede ise Sovyet marşının söz yazarı Sergey Mihalkov'un yeniden yazdığı sözler kullanılmıştır. Sovyetler Birliği Marşı, 1944 yılından itibaren "Enternasyonal" marşının yerine resmî marş olarak kullanılmaya başlandı. Marş, 1956-1977 yılları arasında destalinizasyon politikaları nedeniyle sözsüz, sadece enstrümantal çalındı. 1977 yılında eski Sovyetler Birliği Komünist Partisi Genel Sekreteri Josef Stalin'e atıf yapan sözler çıkartıldı, yine Mihalkov tarafından yazılan komünizmin ve II. Dünya Savaşı'nın zaferlerine vurgu yapan sözlerle değiştirilerek kullanılmaya başlandı.

Rusya Sovyet Federatif Sosyalist Cumhuriyeti dışındaki Sovyetler Birliği'nin tüm cumhuriyetlerinin kendi ulusal marşları bulunmaktaydı. Mihail Glinka'nın bestelediği sözsüz "Patriotiçeskaya Pesnya" 1990 yılında Rusya Yüksek Sovyeti tarafından resmî olarak kabul edildi ve Sovyetler Birliği'nin dağılmasından sonra, 1993 yılında Rusya devlet başkanı Boris Yeltsin tarafından resmen yürürlüğe kondu. Marşın kamuoyunda pek bilinmemesi, politikacılar arasında kullanılmaması, uluslararası yarışmalarda Rusyalı oyuncular tarafından ilhamla okunmaması gibi sebeplerle uzun süre beğenilmedi ve popüler olmadı. Hükûmet, yeni şarkı sözleri oluşturmak üzere gerçekleştirilen pek çok yarışmayı destekledi ancak hiçbir söz resmî olarak kabul edilmedi.

Yeltsin'den sonra devlet başkanı olan Vladimir Putin'in 7 Mayıs 2000 tarihinde göreve başlamasından kısa süre sonra "Patriotiçeskaya Pesnya" marşı değiştirildi. Federal Meclis, Sovyetler Birliği Marşı'nın müziği üzerine yazılan yeni sözlerle oluşturulan yeni marşı Aralık ayında onayladı. Böylece yeni Rusya Marşı, Sovyetler Birliği'nin dağılmasından sonra Rusya tarafından kullanılan ikinci marş oldu. Hükûmet, yeni sözler yazılması için kampanyalar düzenledi. Sonunda Sergey Mihalkov tarafından yazılan ve Rusya tarihi ile eski tarihi geleneklere değinen sözler kabul edildi.

Marşın kamuoyunda algılanması ve Rus halkında kabulü tartışmalıdır. Bazı kesimlere göre yeni marş Rusya'nın en iyi günlerini ve geçmiş fedakârlıkları hatırlatırken çeşitli eleştirmenlere göre Stalin liderliğindeki yönetim dönemindeki şiddet olaylarını hatırlatmaktadır. Hükûmet ise marşın halkların birliğinin simgesi olduğunu ve geçmişe saygılı olduğunu savunmaktadır. 2009'da yapılan bir ankete göre katılımcıların %56'sı marşı dinlerken gurur duyduğunu ifade etmiştir.
"""

summary_template_1 = """
Please provide a detailed summary of the following text in English:
{text}
"""

summary_template_2 = """
Please make comments and give more information about the following text in English:
{text}
"""

summary_template_3 = """
Please translate the text to Spanish:
{text}
"""

template_1 = PromptTemplate(input_variables=["text"], template=summary_template_1)
template_2 = PromptTemplate(input_variables=["text"], template=summary_template_2)
template_3 = PromptTemplate(input_variables=["text"], template=summary_template_3)

llm = ChatOllama(model="llama3.1")


chain_1 = template_1 | llm | StrOutputParser()
chain_2 = template_2 | llm | StrOutputParser()
chain_3 = template_3 | llm | StrOutputParser()

def custom_prompt_selector(chains):
    return random.choice(chains)


def summarize_text(text):
    selected_chain = custom_prompt_selector([chain_1, chain_2, chain_3])
    
    summary = selected_chain.invoke({"text": text})
    
    return summary

if __name__ == "__main__":
    print("Starting text summarization application with LLaMA...")

    summary = summarize_text(text_to_summarize)

    print("Generated Summary:")
    print(summary)
