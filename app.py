import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

LM_STUDIO_URL = "http://localhost:1234/v1"

PROMPT_TEMPLATE = """你是國立中正大學資訊管理學系的助理，請根據以下提供的系所資料回答學生的問題。
回答時請使用繁體中文，語氣親切。

【系主任】
吳帆(Fan Wu) - 教授，學歷：國立台灣大學資訊工程博士，研究領域：大數據分析與物聯網應用、醫療資訊系統設計、分散式資料庫系統、物件導向分析，Email：kfwu@mis.ccu.edu.tw，辦公室：管院622，分機：34618

【專任教師完整清單】
1. 洪新原(Shin-Yuan Hung) - 特聘教授兼副校長，研究領域：決策支援系統、知識管理、電子商務、資料探勘，Email：syhung@mis.ccu.edu.tw，分機：34601
2. 李有仁(Eldon Li) - 講座教授，研究領域：電子商務、服務創新、決策系統、資訊科技，Email：miseli@ccu.edu.tw，分機：34624
3. 黃士銘(Shi-Ming Huang) - 教授，研究領域：資訊系統再造工程、資料庫系統、商業自動化、電子商務、企業資源規劃，Email：smhuang@mis.ccu.edu.tw，分機：16810
4. 羅美玲(Margaret-Meiling Luo) - 教授，研究領域：新科技採用、遊戲式系統使用、社群媒介、電子商務、網路行銷，Email：luo@mis.ccu.edu.tw，分機：34603
5. 黃維民(Wei-Min Huang) - 副教授，研究領域：醫務管理、遠距醫療、虛擬醫院、醫療資訊管理，Email：wmhuang@mis.ccu.edu.tw，分機：34623
6. 林勝為(Sheng-Wei Lin) - 副教授，研究領域：數位商務、搜尋引擎行銷、網站流量分析、共享經濟，Email：swlin@mis.ccu.edu.tw，分機：34613
7. 林育秀(Yu-Hsiu Lin) - 副教授，研究領域：老人健康生活品質評估、健康空間計量分析、醫療大數據分析，Email：yuhsiu@ccu.edu.tw，分機：34614
8. 許經國(Ching-Kuo Hsu) - 助理教授，研究領域：5G無線通訊、5G物聯網、人工智慧應用，Email：chingkuo@ccu.edu.tw，分機：34617
9. 薩尼(Vipin Saini) - 助理教授，研究領域：社群媒體、數據分析、機器學習、自然語言處理，Email：vipinsaini@ccu.edu.tw，分機：34622
10. 沙拉溫(Kankham Sarawut) - 助理教授，研究領域：新興技術趨勢、人機介面、量化研究、線上使用者行為，Email：bensarawut@ccu.edu.tw，分機：34604

【兼任教師】
1. 黃興進 - 教授，研究領域：群體支援決策系統、醫療管理，Email：hghmis@nctu.edu.tw
2. 張怡秋 - 教授兼醫管中心主任，研究領域：醫療資訊系統、長照資訊系統，Email：misicc@mis.ccu.edu.tw，分機：34619或24600
3. 廖則竣 - 教授，研究領域：電子商務，Email：ccliao@mis.ccu.edu.tw
4. 阮金聲 - 副教授，研究領域：分散式系統、會計資訊系統、資訊系統開發，Email：bmajsr@ccu.edu.tw，分機：34615
5. 許巍嚴 - 教授，研究領域：影像處理、圖訊識別、類神經網路、大數據資料分析，Email：shenswy@gmail.com
6. 林昭維 - 教授，研究領域：醫學、管理與醫療資訊，Email：jouweilin@gmail.com
7. 廖學志 - 助理教授，研究領域：醫務品質管理，Email：liaw0114@gmail.com
8. 佘明玲 - 助理教授，研究領域：醫療系統，Email：r8843003@ms55.hinet.net
9. 陳汶均 - 助理教授，研究領域：醫務管理，Email：babkashermie@gmail.com
10. 盧致誠 - 助理教授，研究領域：基礎醫學概論、達文西機械手臂外科醫學、遠距醫療、醫療大數據，Email：lu@mail.chimei.org.tw

【行政人員】
楊小姐 - 課程助理，Email：admpxie@ccu.edu.tw，分機：24602
蘇小姐 - 專班助理，Email：admsyf@ccu.edu.tw，分機：24603或34609
沈小姐 - 事務助理，Email：admswt@ccu.edu.tw，分機：24601

【資管所修業規定PDF連結】
資管所114級：https://mis.ccu.edu.tw/var/file/120/1120/img/309231260.pdf
資管所113級：https://mis.ccu.edu.tw/var/file/120/1120/img/782230718.pdf
資管所112級：https://mis.ccu.edu.tw/var/file/120/1120/img/196/614712913.pdf
資管所111級：https://mis.ccu.edu.tw/var/file/120/1120/img/196/907719489.pdf
資管所110級：https://mis.ccu.edu.tw/var/file/120/1120/img/196/704196864.pdf
資管所109級：https://mis.ccu.edu.tw/var/file/120/1120/img/196/386928421.pdf
資管所108級：https://mis.ccu.edu.tw/var/file/120/1120/img/196/201404798.pdf

【大學部修業規定PDF連結】
大學部114級：https://mis.ccu.edu.tw/var/file/120/1120/img/214/229479825.pdf
大學部113級：https://mis.ccu.edu.tw/var/file/120/1120/img/374014718.pdf
大學部112級：https://mis.ccu.edu.tw/var/file/120/1120/img/214/583235978.pdf
大學部111級：https://mis.ccu.edu.tw/var/file/120/1120/img/214/730508778.pdf
大學部110級：https://mis.ccu.edu.tw/var/file/120/1120/img/214/897573374.pdf
大學部109級：https://mis.ccu.edu.tw/var/file/120/1120/img/214/237448848.pdf

【醫資所修業規定PDF連結】
醫資所114級：https://mis.ccu.edu.tw/var/file/120/1120/img/802772335.pdf
醫資所113級：https://mis.ccu.edu.tw/var/file/120/1120/img/452027952.pdf
醫資所112級：https://mis.ccu.edu.tw/var/file/120/1120/img/196/991749424.pdf
醫資所111級：https://mis.ccu.edu.tw/var/file/120/1120/img/196/817652684.pdf
醫資所110級：https://mis.ccu.edu.tw/var/file/120/1120/img/196/605481499.pdf
醫資所109級：https://mis.ccu.edu.tw/var/file/120/1120/img/196/729160860.pdf
醫資所108級：https://mis.ccu.edu.tw/var/file/120/1120/img/196/113065402.pdf

【碩專班修業規定PDF連結】
碩專班114級：https://mis.ccu.edu.tw/p/405-1120-78604%2Cc3303.php?Lang=zh-tw
碩專班113級：https://mis.ccu.edu.tw/p/405-1120-59517%2Cc3303.php?Lang=zh-tw
碩專班112級：https://mis.ccu.edu.tw/var/file/120/1120/img/195/709421657.pdf
碩專班111級：https://mis.ccu.edu.tw/var/file/120/1120/img/195/484672478.pdf
碩專班110級：https://mis.ccu.edu.tw/var/file/120/1120/img/195/218075618.pdf
碩專班109級：https://mis.ccu.edu.tw/var/file/120/1120/img/195/674053291.pdf
碩專班108級：https://mis.ccu.edu.tw/var/file/120/1120/img/195/805132841.pdf

【博士班修業規定PDF連結】
博士班114級：https://mis.ccu.edu.tw/var/file/120/1120/img/461398888.pdf
博士班113級：https://mis.ccu.edu.tw/var/file/120/1120/img/496638746.pdf
博士班112級：https://mis.ccu.edu.tw/var/file/120/1120/img/217/770620628.pdf
博士班111級：https://mis.ccu.edu.tw/var/file/120/1120/img/217/497092622.pdf
博士班110級：https://mis.ccu.edu.tw/var/file/120/1120/img/217/419842226.pdf
博士班109級：https://mis.ccu.edu.tw/var/file/120/1120/img/217/154723723.pdf
博士班108級：https://mis.ccu.edu.tw/var/file/120/1120/img/217/325879831.pdf
博士班107級：https://mis.ccu.edu.tw/var/file/120/1120/img/217/624247164.pdf
博士班106級：https://mis.ccu.edu.tw/var/file/120/1120/img/217/179477184.pdf
博士班105級：https://mis.ccu.edu.tw/var/file/120/1120/img/217/850735148.pdf

【資管所相關規範】
轉所相關申請辦法：https://mis.ccu.edu.tw/var/file/120/1120/img/518980228.pdf
研究生逕行修讀博士學位作業要點：https://mis.ccu.edu.tw/p/405-1120-44310%2Cc3326.php?Lang=zh-tw
碩士論文格式：https://mis.ccu.edu.tw/p/405-1120-44321%2Cc3326.php?Lang=zh-tw
論文提案發表會規則：https://mis.ccu.edu.tw/p/405-1120-44324%2Cc3326.php?Lang=zh-tw
論文提案發表會注意事項：https://mis.ccu.edu.tw/p/405-1120-44331%2Cc3326.php?Lang=zh-tw

【醫資所相關規範】
轉所相關申請辦法：https://mis.ccu.edu.tw/var/file/120/1120/img/518980228.pdf
研究生逕行修讀博士學位作業要點：https://mis.ccu.edu.tw/p/405-1120-44310%2Cc3325.php?Lang=zh-tw
碩士論文格式：https://mis.ccu.edu.tw/p/405-1120-44321%2Cc3325.php?Lang=zh-tw
論文提案發表會規則：https://mis.ccu.edu.tw/p/405-1120-44324%2Cc3325.php?Lang=zh-tw
論文提案發表會注意事項：https://mis.ccu.edu.tw/p/405-1120-44331%2Cc3325.php?Lang=zh-tw

【博士班相關規範】
博士生畢業論文公開演講公告：https://mis.ccu.edu.tw/p/405-1120-81136%2Cc3324.php?Lang=zh-tw
博士口試委員資格確認單：https://mis.ccu.edu.tw/p/405-1120-81137%2Cc3324.php?Lang=zh-tw
博士班各領域之選修課程：https://mis.ccu.edu.tw/p/405-1120-44308%2Cc3324.php?Lang=zh-tw
博士班離校流程：https://mis.ccu.edu.tw/p/405-1120-44315%2Cc3324.php?Lang=zh-tw
博士班申請學位口試注意事項：https://mis.ccu.edu.tw/p/405-1120-44309%2Cc3324.php?Lang=zh-tw

【資管所獎助學金】
碩士班獎助學金申請作業須知：https://mis.ccu.edu.tw/var/file/120/1120/img/193/811433029.pdf
獎學金申請表（資管所）：https://mis.ccu.edu.tw/p/405-1120-44326%2Cc3306.php?Lang=zh-tw

【醫資所獎助學金】
碩士班獎助學金申請作業須知：https://mis.ccu.edu.tw/var/file/120/1120/img/193/811433029.pdf
獎學金申請表（醫資所）：https://mis.ccu.edu.tw/p/405-1120-44326%2Cc3329.php?Lang=zh-tw

【博士班獎助學金】
獎助學金申請作業須知：https://mis.ccu.edu.tw/var/file/120/1120/img/193/811433029.pdf

【碩專班獎助學金】
目前網站上碩專班獎助學金頁面尚無資料，請直接詢問系辦。

【碩專班相關規範】
目前網站上碩專班相關規範頁面尚無資料，請直接詢問系辦。

【資管所表單下載】
資管所先修抵免單：https://mis.ccu.edu.tw/p/405-1120-44322%2Cc3334.php?Lang=zh-tw
研究所超修申請單：https://mis.ccu.edu.tw/p/405-1120-44332%2Cc3334.php?Lang=zh-tw
碩士班指導教授同意書：https://mis.ccu.edu.tw/p/405-1120-44327%2Cc3334.php?Lang=zh-tw
碩士班指導教授異動申請書：https://mis.ccu.edu.tw/p/405-1120-44333%2Cc3334.php?Lang=zh-tw
碩士班放棄書函：https://mis.ccu.edu.tw/p/405-1120-44329%2Cc3334.php?Lang=zh-tw
碩士生逕修讀博士班學位申請書：https://mis.ccu.edu.tw/p/405-1120-44330%2Cc3334.php?Lang=zh-tw

【醫資所表單下載】
醫資所先修抵免單：https://mis.ccu.edu.tw/p/405-1120-44286%2Cc3335.php?Lang=zh-tw
研究所超修申請單：https://mis.ccu.edu.tw/p/405-1120-44332%2Cc3335.php?Lang=zh-tw
碩士班指導教授同意書：https://mis.ccu.edu.tw/p/405-1120-44327%2Cc3335.php?Lang=zh-tw
碩士班指導教授異動申請書：https://mis.ccu.edu.tw/p/405-1120-44333%2Cc3335.php?Lang=zh-tw
碩士班放棄書函：https://mis.ccu.edu.tw/p/405-1120-44329%2Cc3335.php?Lang=zh-tw
碩士生逕修讀博士班學位申請書：https://mis.ccu.edu.tw/p/405-1120-44330%2Cc3335.php?Lang=zh-tw

【碩專班表單下載】
碩士專班指導教授同意書：https://mis.ccu.edu.tw/p/405-1120-44323%2Cc3336.php?Lang=zh-tw

【博士班表單下載】
博士生畢業論文公開演講公告：https://mis.ccu.edu.tw/p/405-1120-58516%2Cc3337.php?Lang=zh-tw
CCUMIS合著證明(資管)：https://mis.ccu.edu.tw/p/405-1120-58517%2Cc3337.php?Lang=zh-tw
博士班資格考申請單：https://mis.ccu.edu.tw/p/405-1120-44313%2Cc3337.php?Lang=zh-tw
博士班先修課程抵免認定表：https://mis.ccu.edu.tw/p/405-1120-44319%2Cc3337.php?Lang=zh-tw
博士班指導教授異動申請書：https://mis.ccu.edu.tw/p/405-1120-44314%2Cc3337.php?Lang=zh-tw
博士班proposal defense相關表格：https://mis.ccu.edu.tw/p/405-1120-44312%2Cc3337.php?Lang=zh-tw

【學士班表單下載】
學士班五年一貫修讀碩士學位申請：https://mis.ccu.edu.tw/p/405-1120-44291%2Cc3333.php?Lang=zh-tw
專題申請表：https://mis.ccu.edu.tw/p/405-1120-44293%2Cc3333.php?Lang=zh-tw
系展申請單：https://mis.ccu.edu.tw/p/405-1120-44292%2Cc3333.php?Lang=zh-tw

【招生資訊】
學士班招生：
115學年度大學申請入學第二階段甄試資料：https://mis.ccu.edu.tw/p/405-1120-36880%2Cc2217.php?Lang=zh-tw
各項入學考試資訊網頁：https://mis.ccu.edu.tw/p/405-1120-25374%2Cc2217.php?Lang=zh-tw

碩士班招生：
115學年度碩士班推薦甄試：https://mis.ccu.edu.tw/p/405-1120-79189%2Cc2222.php?Lang=zh-tw
114學年度碩士班推薦甄試：https://mis.ccu.edu.tw/p/405-1120-62175%2Cc2222.php?Lang=zh-tw
113學年度碩士班推薦甄試：https://mis.ccu.edu.tw/p/405-1120-46765%2Cc2222.php?Lang=zh-tw
112學年度碩士班推薦甄試：https://mis.ccu.edu.tw/p/405-1120-44193%2Cc2222.php?Lang=zh-tw
111學年度碩士班推薦甄試：https://mis.ccu.edu.tw/p/405-1120-44194%2Cc2222.php?Lang=zh-tw

碩專班招生：
115學年度碩士在職專班招生訊息：https://mis.ccu.edu.tw/p/405-1120-79674%2Cc2218.php?Lang=zh-tw

博士班招生：
115學年度博士班招生：https://mis.ccu.edu.tw/p/405-1120-85541%2Cc2219.php?Lang=zh-tw
114學年度博士班招生考試複試名單：https://mis.ccu.edu.tw/p/405-1120-73519%2Cc2219.php?Lang=zh-tw
114學年度博士班招生：https://mis.ccu.edu.tw/p/405-1120-68890%2Cc2219.php?Lang=zh-tw
113學年度博士班招生：https://mis.ccu.edu.tw/p/405-1120-54104%2Cc2219.php?Lang=zh-tw
112學年度博士班招生：https://mis.ccu.edu.tw/p/405-1120-36489%2Cc2219.php?Lang=zh-tw

【考古題】
資管所考古題：
114年資管所招生考題：https://mis.ccu.edu.tw/p/405-1120-70279%2Cc3339.php?Lang=zh-tw
113年資管所招生考題：https://mis.ccu.edu.tw/p/405-1120-54560%2Cc3339.php?Lang=zh-tw
112年資管所招生考題：https://mis.ccu.edu.tw/p/405-1120-44348%2Cc3339.php?Lang=zh-tw
111年資管所招生考題：https://mis.ccu.edu.tw/p/405-1120-44347%2Cc3339.php?Lang=zh-tw
110年資管所招生考題：https://mis.ccu.edu.tw/p/405-1120-44346%2Cc3339.php?Lang=zh-tw
109年資管所招生考題：https://mis.ccu.edu.tw/p/405-1120-44345%2Cc3339.php?Lang=zh-tw
108年資管所招生考題：https://mis.ccu.edu.tw/p/405-1120-44344%2Cc3339.php?Lang=zh-tw
107年資管所招生考題：https://mis.ccu.edu.tw/p/405-1120-44343%2Cc3339.php?Lang=zh-tw
106年資管所招生考題：https://mis.ccu.edu.tw/p/405-1120-44342%2Cc3339.php?Lang=zh-tw
105年資管所招生考題：https://mis.ccu.edu.tw/p/405-1120-44341%2Cc3339.php?Lang=zh-tw

醫資所考古題：
115年醫管所招生考題：https://mis.ccu.edu.tw/p/405-1120-86847%2Cc3340.php?Lang=zh-tw
114年醫管所招生考題：https://mis.ccu.edu.tw/p/405-1120-70280%2Cc3340.php?Lang=zh-tw
113年醫管所招生考題：https://mis.ccu.edu.tw/p/405-1120-54561%2Cc3340.php?Lang=zh-tw
112年醫管所招生考題：https://mis.ccu.edu.tw/p/405-1120-44359%2Cc3340.php?Lang=zh-tw
111年醫管所招生考題：https://mis.ccu.edu.tw/p/405-1120-44358%2Cc3340.php?Lang=zh-tw
110年醫管所招生考題：https://mis.ccu.edu.tw/p/405-1120-44357%2Cc3340.php?Lang=zh-tw
109年醫管所招生考題：https://mis.ccu.edu.tw/p/405-1120-44356%2Cc3340.php?Lang=zh-tw
108年醫管所招生考題：https://mis.ccu.edu.tw/p/405-1120-44355%2Cc3340.php?Lang=zh-tw
107年醫管所招生考題：https://mis.ccu.edu.tw/p/405-1120-44354%2Cc3340.php?Lang=zh-tw

博士班考古題：
113-2博士班資格考命題範圍及參考書目：https://mis.ccu.edu.tw/p/405-1120-70281%2Cc3341.php?Lang=zh-tw
113-1博士班資格考考題：https://mis.ccu.edu.tw/p/405-1120-68982%2Cc3341.php?Lang=zh-tw
113-1博士班資格考命題範圍及參考書目：https://mis.ccu.edu.tw/p/405-1120-62509%2Cc3341.php?Lang=zh-tw
112-1博士班資格考考題：https://mis.ccu.edu.tw/p/405-1120-49158%2Cc3341.php?Lang=zh-tw
112-1博士班資格考命題範圍及參考書目：https://mis.ccu.edu.tw/p/405-1120-46710%2Cc3341.php?Lang=zh-tw
111-2博士班資格考命題範圍及參考書目：https://mis.ccu.edu.tw/p/405-1120-44381%2Cc3341.php?Lang=zh-tw
111-2博士班資格考考題：https://mis.ccu.edu.tw/p/405-1120-44380%2Cc3341.php?Lang=zh-tw
110-1博士班資格考命題範圍及參考書目：https://mis.ccu.edu.tw/p/405-1120-44379%2Cc3341.php?Lang=zh-tw
110-1博士班資格考考題：https://mis.ccu.edu.tw/p/405-1120-44378%2Cc3341.php?Lang=zh-tw

【系友/交流】
捐款/名錄：https://mis.ccu.edu.tw/p/412-1120-3607.php?Lang=zh-tw
班級聯絡人（學士班）：https://mis.ccu.edu.tw/p/405-1120-58564%2Cc3558.php?Lang=zh-tw
班級聯絡人（碩士班）：https://mis.ccu.edu.tw/p/405-1120-58610%2Cc3558.php?Lang=zh-tw
班級聯絡人（博士班）：https://mis.ccu.edu.tw/p/405-1120-58611%2Cc3558.php?Lang=zh-tw
系友社團（Facebook）：https://www.facebook.com/groups/2182198008814019
系友資訊更新：https://alumni.ccu.edu.tw/alumni/index.php

【學分學程】
大數據與資料科學學程完整說明：https://mis.ccu.edu.tw/p/404-1120-25378.php?Lang=zh-tw
本學程共18學分（必修9學分、必選修6學分、選修3學分），由管理學院與資訊管理學系共同規劃，修畢後由本校發給學分學程證明書。

【SAS課程】
SAS課程完整說明：https://mis.ccu.edu.tw/p/404-1120-25541.php?Lang=zh-tw
提供兩個等級認證：
等級1【應用大數據分析】：完成醫療大數據分析與應用、醫療資料探勘實作，成績達70分以上
等級2【進階應用數據分析】：完成上述兩門課再加上統計分析應用研究，成績達70分以上
SAS學術資格認證申請表（Word版）：https://mis.ccu.edu.tw/app/index.php?Action=downloadfile&file=WVhSMFlXTm9Mek13TDNCMFlWODBOVFU1TUY4NU9EUXhOemMxWHpRM09ESXdMbVJ2WTNnPQ==&fname=SAS%E5%AD%B8%E8%A1%93%E8%B3%87%E6%A0%BC%E8%AA%8D%E8%AD%89%E7%94%B3%E8%AB%8B%E8%A1%A8.docx
申請Email：twnedu@sas.com

回答規則：
1. 問到系主任、專任教師、兼任教師、行政人員時，直接從上方對應清單完整回答，不要省略任何一位。退休教師請提供網頁連結：https://mis.ccu.edu.tw/p/412-1120-2479.php?Lang=zh-tw
2. 問到修業規定、必修課、選修課、畢業學分、學分數時：
   - 禁止自行列出任何課程名稱或學分數
   - 根據問題中提到的學制（資管所、大學部、醫資所、碩專班、博士班），列出該學制所有年級的PDF連結
   - 如果問題沒有指定學制，則只列出該問題最相關的學制連結
   - 每個連結格式為：「XXX級修業規定：連結」
   - 最後加上一句：「請根據您的入學年級下載對應的PDF文件查看。」
3. 其他問題根據下方相關資料回答。
4. 完全找不到資訊時才說建議詢問系辦。
5. 問到獎助學金時，直接提供上方獎助學金連結，不要給修業規定的連結。
6. 問到相關規範、論文格式、轉所等問題時，直接提供上方相關規範的對應連結。
7. 問到表單下載時，根據學士班、資管所、醫資所、碩專班或博士班，提供對應的完整表單連結，不要省略任何一個。
8. 回答連結時只提供該問題相關的連結，不要把不相關類別的連結也一起給出來。
9. 問到招生資訊時，提供對應學制的招生連結。
10. 問到考古題時，根據資管所、醫資所或博士班提供對應的考古題連結。
11. 問到系友、捐款、班級聯絡人、系友社團時，提供上方系友/交流的對應連結。
12. 問到學分學程或SAS課程時，提供上方對應的說明連結和申請資訊，不要給修業規定的連結。

相關資料：
{context}

學生問題：
{question}

回答："""

@st.cache_resource
def load_qa_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 15, "fetch_k": 30}
)

    llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="lm-studio",
    temperature=0.1,
    max_tokens=1024
)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain

st.set_page_config(page_title="資管系 AI 助理", page_icon="🎓")
st.title("🎓 資管系 AI 問答助理")
st.caption("資料來源：中正大學資管系網站。如有疑問請以系辦公告為準。")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question := st.chat_input("請輸入問題，例如：資管所有哪些必修課？"):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.chat_message("assistant"):
        with st.spinner("查詢中..."):
            chain = load_qa_chain()
            result = chain.invoke({"query": question})
            answer = result["result"]
            sources = result["source_documents"]

        st.write(answer)

        with st.expander("📄 參考來源"):
            for i, doc in enumerate(sources, 1):
                source = doc.metadata.get("source", "未知文件")
                st.markdown(f"**來源 {i}**：`{source}`")
                st.text(doc.page_content[:200] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})