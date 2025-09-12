from pathlib import Path
import re
from typing import Dict, Any, List, Optional
from jinja2 import Template

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field

from config import get_openai_llm
from tools.tbi_tool import TBICalculator, TBICalculatorFactory, S3Client
from tools.data_validator_tool import DataValidator


def _read_prompt() -> str:
    """Read and return the Jinja2 template content"""
    base_dir = Path(__file__).resolve().parents[1]
    prompt_path = base_dir / "prompts" / "tbi_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")

# "tci_1": "Đây là thời điểm thiết lập nền tảng cho toàn bộ quá trình giao dịch sau này. Đây là lúc trader bắt đầu hình thành cách phản ứng với rủi ro, mức độ kỷ luật và thói quen quản trị cảm xúc. Những trải nghiệm sớm dù là thua lỗ hay thành công đều là dữ liệu quan trọng, định hình niềm tin, chiến lược tư duy và mức độ kiên định khi đối diện với thị trường.",
# "tci_2": "Đây là thời điểm trader bắt đầu bước vào sự ổn định trong tư duy và chiến lược giao dịch. Đây là lúc bạn có sự chuyển biến rõ rệt về nhận thức, quan điểm và cách tiếp cận thị trường. Sự trưởng thành trong giai đoạn này thể hiện qua việc kiểm soát cảm xúc tốt hơn, kiên định với kế hoạch, và dần xây dựng phong cách giao dịch bền vững.",
# "tci_3": "Đây là giai đoạn nhìn lại toàn bộ hành trình đã đi qua để nhận diện rõ điểm mạnh, điểm yếu và rút ra bài học từ trải nghiệm thực chiến. Giai đoạn này giúp bạn xác định đâu là phong cách giao dịch, thị trường và khung thời gian phù hợp nhất với DNA bản chất của mình, từ đó vạch ra hướng đi chiến lược lâu dài. Quan trọng hơn, đây là lúc bạn phân biệt được giữa đam mê thật sự và những áp lực đến từ kỳ vọng bên ngoài, để từng bước định vị bản đồ giao dịch cá nhân và chuẩn bị cho một giai đoạn phát triển bền vững, tự chủ và nhất quán hơn.",
# "tci_4": "Đây là giai đoạn bạn vừa có cơ hội cống hiến, lan tỏa giá trị tích cực cho cộng đồng, vừa tận hưởng thành quả và sự tự do mà mình đã gây dựng. Để giữ được sự cân bằng này, bạn cần nền tảng kỷ luật, tri thức và kinh nghiệm đã rèn luyện từ những chặng trước. Đây chính là lúc bạn giao thoa giữa sứ mệnh phụng sự và niềm vui sống trọn vẹn, biến hành trình của mình thành một minh chứng cho sự trưởng thành bền vững.",

tbi_definitions = {
    "edi": "Emotional Drive Index - chỉ số này phản ánh khát khao hành vi sâu thẳm chi phối quyết định giao dịch. Nó lý giải cách trader phản ứng với thị trường và áp lực cảm xúc.",
    "ppai": "Path Potential Alignment Index - đo lường mức độ liên kết giữa hành trình hành vi gốc (Path) và vai trò tiềm năng cần đạt (Potential). Chỉ số này cho trader biết những hành vi nào cần rèn luyện và điều chỉnh để vừa tốt nghiệp được bài học hành vi cốt lõi, vừa hoàn thành sứ mệnh giao dịch và tiến hóa thành phiên bản Elite Trader.",
    "spi": "Skill Potential Index - cho thấy vai trò hành vi cốt lõi mà một trader cần phát huy để đạt đỉnh cao trong giao dịch. Khi bạn hoàn thiện chỉ số này bạn sẽ vừa thỏa mãn đam mê, vừa tạo ra tác động tích cực trong cộng đồng trader. Chỉ số `SPI` không chỉ phản ánh điểm mạnh hiện tại, mà còn chỉ ra năng lực tiềm ẩn cần khai thác để trở thành Elite Trader.",
    "cmi": "Crisis Management Index - phản ánh cách trader ứng phó với khó khăn và áp lực trong giao dịch. Chỉ số này cho biết khả năng giữ vững sự tỉnh táo, phân tích tình huống và lựa chọn hành động đúng đắn khi thị trường biến động.",
    "mpi": "Market Persona Index - phản ánh cách trader được thị trường và cộng đồng nhìn nhận thông qua hành vi, năng lượng và phong cách giao dịch mà họ thể hiện ra ngoài. Chỉ số này giúp trader phát đi `tín hiệu tính cách` tới thế giới với sự tự tin, thận trọng, sáng tạo hay quyết đoán và giúp họ nhận biết mức độ nhất quán giữa bản chất bên trong và hình ảnh bên ngoài.",
    "ri": "Resilience Index - phản ánh giai đoạn bạn đạt độ chín trong tư duy và sức bền giao dịch. Chỉ số này thể cho bạn biết thời điểm năng lượng, trải nghiệm và khả năng kiểm soát rủi ro được phát huy mạnh mẽ nhất. Chỉ số `RI` tập trung vào chiến lược, kỷ luật và quản trị vốn để tối ưu hiệu suất và xây dựng sự bền vững dài hạn trong hành trình trading.",
    "ioci": "Inner Outer Coherence Index - là chỉ số đo lường mức độ hòa hợp giữa động lực nội tâm và cách bạn thể hiện ra bên ngoài trong giao dịch. Chỉ số `IOCI` đóng vai trò như chiếc cầu nối giữa cách bạn nhìn nhận bản thân và hình ảnh mà thị trường, cộng đồng thấy ở bạn. `IOCI` giúp trader nhận ra sự khác biệt giữa “tôi thật sự là ai” và “tôi đang thể hiện như thế nào”, từ đó đưa ra điều chỉnh để duy trì sự nhất quán, giảm hiểu lầm và củng cố niềm tin.",
    "tai": "Trading Attitude Index - phản ánh thái độ và góc nhìn cốt lõi mà trader mang vào thị trường. Chỉ số `TAI` cho thấy cách bạn tiếp nhận tình huống, cơ hội và rủi ro trong từng giai đoạn giao dịch. Chỉ số này giúp bạn chủ động điều chỉnh thái độ để duy trì kỷ luật, tập trung và đón nhận giá trị tích cực từ thị trường.",
    "ppa": "Path Potential Alignment - là chỉ số cốt lõi quan trọng nhất trong bản đồ hành vi giao dịch, chiếm tới 50% đến 60% khả năng thành công dài hạn của một trader. Chỉ số `PPA` phản ánh con đường phát triển tự nhiên của bạn trong trading: mục tiêu, phong cách hành vi nổi bật, những rào cản thường gặp và “bài học lớn” bạn cần vượt qua để nâng cấp bản thân. Khi bạn duy trì sự đồng bộ giữa `PPA` và chiến lược giao dịch cá nhân, bạn sẽ đạt đến trạng thái ổn định, kỷ luật và bền vững. Ngược lại, khi lệch khỏi `PPA`, bạn dễ rơi vào vòng lặp cảm tính, thiếu định hướng và dễ bỏ cuộc.",
    "wmi": "Weakness Map Index - phản ánh những hành vi giao dịch và năng lực tâm lý mà trader chưa được trang bị bẩm sinh khi bước vào thị trường. Việc nhận diện rõ những thiếu hụt từ chỉ số này sẽ giúp trader biết mình cần rèn luyện ở đâu, phát triển phẩm chất gì để lấp đầy lỗ hổng hành vi.",
    "ssi": "Subconscious Stability Index - phản ánh nền tảng hành vi vô thức mà trader cần rèn luyện để dễ dàng đạt được mục tiêu giao dịch, cũng như giữ vững sự ổn định khi đối mặt với áp lực thị trường và chuỗi biến động bất lợi. Đây là thước đo cho thấy độ bền tâm lý ngầm và cách tiềm thức của bạn phản ứng trong những tình huống căng thẳng, từ đó quyết định khả năng duy trì kỷ luật và phục hồi sau rủi ro.",
    "sai": "Strength Amplifier Index - phản ánh nguồn năng lượng tiềm ẩn mà trader thường vô thức dựa vào khi đưa ra quyết định. Đây là `mạch ngầm` thúc đẩy hành vi và có thể trở thành siêu sức mạnh nếu được khai thác đúng, nhưng cũng là cạm bẫy nếu không được kiểm soát.",
    "nei": "Natural Edge Index - được xem như lợi thế tự nhiên mà mỗi trader sở hữu, chỉ số này tập trung phản ánh bản năng, kỹ năng bẩm sinh và nguồn sức mạnh hành vi đặc trưng ngay từ khi bắt đầu giao dịch. Đây là `edge tự nhiên` cho thấy bạn dễ dàng nổi bật ở khía cạnh nào trong quá trình ra quyết định: quan sát, tốc độ, sự kiên nhẫn, hay khả năng dẫn dắt. Từ chỉ số này, trader sẽ hình dung rõ môi trường giao dịch phù hợp nhất để phát huy lợi thế bẩm sinh, cũng như biết cách biến nó thành điểm tựa chiến lược trong hành trình trở thành Elite Trader.",
    "bli": "Behavioral Liability Index - phản ánh những `bài học` mà trader cần nhận diện và vượt qua trong quá trình phát triển. Đây là những thử thách tiềm ẩn trong tư duy và cảm xúc, thường khiến bạn dễ mắc kẹt hoặc lặp lại sai lầm nếu thiếu kỷ luật.",
    "ari": "Analytical Reasoning Index - phản ánh cách bạn phân tích dữ liệu và định hình quyết định giao dịch. Về cốt lõi, chỉ số này cho thấy bạn xử lý thông tin, đánh giá rủi ro và đưa ra lựa chọn như thế nào trong những tình huống thị trường biến động mạnh và áp lực gia tăng. Đây là thước đo quan trọng để nhận diện mức độ logic và tính hệ thống trong toàn bộ quá trình giao dịch.",
    "tci": "Trading Cycle Index - phản ánh những giai đoạn quan trọng trong hành trình trader, nơi năng lượng, kỹ năng và trải nghiệm của bạn đạt tới đỉnh cao và được thử thách mạnh mẽ nhất. Mỗi chu kỳ TCI chỉ ra khoảnh khắc bạn cần tập trung tối đa để chuyển hóa bản thân, từ việc rèn luyện kỷ luật, nâng cao năng lực phân tích cho tới thay đổi cách tiếp cận thị trường.",
    "mri": "Monthly Rhythm Index - phản ánh nhịp độ hành vi và năng lượng trong từng giai đoạn 30 ngày, giúp trader xác định đâu là thời điểm nên mở rộng hành động, đâu là lúc cần chậm lại để củng cố hệ thống. Chỉ số `MRI` cho bạn gợi ý thực tế về cách tập trung và ưu tiên trong tháng, để mỗi bước đi đều mang lại sự tiến bộ và trải nghiệm ý nghĩa hơn trong hành trình giao dịch.",
    "dai": "Daily Alignment Index - phản ánh mức độ phù hợp giữa trạng thái hành vi cá nhân và nhịp thị trường trong từng ngày, từ đó đưa ra gợi ý về việc bạn nên tập trung vào loại hoạt động nào hôm nay `quan sát, học hỏi, review hay hành động quyết đoán`. Nhờ vậy, trader có thể xây dựng kế hoạch giao dịch trong ngày rõ ràng và cân bằng hơn, tối ưu cả kết quả lẫn trải nghiệm.",
    "cii": "Cohort Influence Index - phản ánh ảnh hưởng của bối cảnh thời đại và môi trường thị trường mà trader đang tham gia. Khi hiểu được mối liên kết giữa bản thân và `dòng chảy thế hệ` xung quanh, trader có thể điều chỉnh phong cách giao dịch để giảm xung đột với cộng đồng, thích nghi tốt hơn với xu hướng chung, đồng thời vẫn giữ được bản sắc cá nhân.",
    "ami": "Annual Momentum Index - phản ánh nhịp độ hành vi và dòng năng lượng của trader trong từng năm, giúp bạn dự đoán những thay đổi có thể xuất hiện và xác định trọng tâm phát triển phù hợp cho năm tới. Thay vì chỉ nhìn vào lợi nhuận, AMI cho bạn bản đồ định hướng: năm nay nên tập trung củng cố kỷ luật, học hỏi chiến lược mới, hay mở rộng quy mô giao dịch. Đây là chiếc `la bàn hành vi` giúp bạn đi đúng nhịp với bản thân thay vì bị cuốn theo thị trường."
}


class IndicatorSelection(BaseModel):
    selected_keys: List[str] = Field(description="A list of key behavior analysis indicators most relevant to the user's question.")

def _infer_keys_from_llm(question: str) -> List[str]:
    """
    Use LLM to analyze the semantic question and select the appropriate TBI indicators.
    
    Args:
        question (str): The user's question.
        
    Returns:
        List[str]: The list of selected TBI indicators.
    """
    llm = get_openai_llm()
    parser = JsonOutputParser(pydantic_object=IndicatorSelection)

    # Convert tbi_definitions to string to insert into prompt
    indicator_definitions = "\n".join([f"- **{key}**: {value}" for key, value in tbi_definitions.items()])
    
    # Prompt for LLM
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Bạn là LUMIR AI Behavioral Analyst - Chuyên gia phân tích hành vi giao dịch. 
Nhiệm vụ của bạn là phân tích câu hỏi của người dùng để xác định các chỉ số TBI (Trading Behavior Intelligence) phù hợp và có mối liên hệ với nhau, từ đó đưa ra câu trả lời toàn diện về hành vi giao dịch.

## Các bước phân tích:

1. **Phân tích câu hỏi của người dùng:**
    - Đọc kỹ câu hỏi để xác định vấn đề cốt lõi, các từ khóa chính và các khía cạnh hành vi đang được đề cập (ví dụ: cảm xúc, kỷ luật, chiến lược, thời điểm, quản trị rủi ro, v.v.).
    - Xác định mục tiêu của người dùng khi đặt câu hỏi (ví dụ: hiểu nguyên nhân, tìm giải pháp, đánh giá phong cách).
    
2. **Xác định chỉ số TBI sơ bộ (dựa trên liên kết từ khóa/khái niệm):**
    - Tham chiếu đến danh sách đầy đủ các chỉ số TBI và định nghĩa/giải thích của chúng.
    - Liệt kê tất cả các chỉ số TBI có liên quan trực tiếp hoặc gián tiếp đến các từ khóa và khía cạnh hành vi đã xác định ở Bước 1.

3. **Đảm bảo sự liên kết và toàn diện:**
    - Từ danh sách chỉ số TBI sơ bộ, chọn ra một tập hợp 4-5 chỉ số quan trọng nhất và có mối liên hệ chặt chẽ với nhau.
    - Nguyên tắc lựa chọn: Tính trực tiếp, tính bổ trợ, mối quan hệ nhân-quả hoặc tương quan.

## DANH SÁCH CHỈ SỐ VÀ Ý NGHĨA:

{indicator_definitions}

## QUY TẮC CHỌN CHỈ SỐ:

### **Luôn bắt buộc** (2 chỉ số cốt lõi):
- `ppa`: Chỉ số cốt lõi của mọi phân tích
- `spi`: Chỉ số về điểm mạnh và năng lực tiềm ẩn

### **Chọn thêm dựa trên context**:

#### **1. Vấn đề cảm xúc/tâm lý**:
- `cmi`: Cách phản ứng khi thị trường biến động
- `edi`: Khát khao sâu thẳm trong giao dịch
- `mpi`: Ấn tượng đầu tiên với thị trường

#### **2. Vấn đề hiệu suất và mục tiêu**:
- `ri`: Mức độ bền bỉ, thích ứng qua năm tháng trading
- `ari`: Tư duy lý trí và phân tích thị trường
- `sai`: Vùng sức mạnh đặc biệt giúp trader bứt phá

#### **3. Vấn đề timing và giai đoạn**:
- `mri`: Xu hướng năng lượng trong tháng
- `ami`: Biến động và kỳ vọng trong năm
- `cii`: Ảnh hưởng bối cảnh thế hệ/cohort
- `dai`: Sự phù hợp giữa trạng thái cá nhân và nhịp thị trường trong ngày

#### **4. Vấn đề phát triển và học hỏi**:
- `ssi`: Độ ổn định phản xạ vô thức khi thị trường đảo chiều/áp lực cao
- `wmi`: Kỹ năng và phẩm chất thiếu
- `bli`: Gánh nặng thói quen xấu lặp lại
- `tci`: Giai đoạn phát triển hành vi giao dịch theo tuổi tác

#### **5. Vấn đề thách thức và vượt khó**:
- `tai`: Cách phản ứng và xử lý tình huống hàng ngày
- `nei`: Lợi thế tự nhiên khi giao dịch
- `ppai`: Mức ăn khớp giữa `ppa` và `spi` → đo xác suất bền vững của phong cách giao dịch đã chọn
- `ioci`: Độ nhất quán giữa `edi` và `mpi` trong quyết định giao dịch → giảm xung đột nội tâm, tăng tính mạch lạc.
- `bci`: Thách thức tương ứng với giai đoạn hiện tại

### **Ví dụ phân tích chi tiết**:

#### **"Tôi buồn quá, trading thua lỗ"**:
- `ppa`, `spi` (bắt buộc)
- `cmi` (kiểm soát cảm xúc khi thua lỗ)
- `edi` (động cơ đằng sau quyết định)
- `wmi` (kỹ năng và phẩm chất thiếu)

#### **"Giai đoạn này tôi nên làm gì?"**:
- `ppa`, `spi` (bắt buộc)
- `mri` (xu hướng tháng hiện tại)
- `ri` (chiến lược đỉnh phong độ)
- `cii` (hòa hợp với xu hướng)

#### **"Ưu và nhược điểm khi trading của tôi là gì?"**:
- `ppa`, `spi` (bắt buộc)
- `mpi` (điểm mạnh và điểm yếu)
- `cmi` (khả năng kiểm soát)
- `sai` (thế mạnh riêng)
- `wmi` (điểm yếu cần cải thiện)
- `bli` (khoảng trống kỹ năng)

#### **"Tôi cảm thấy không tự tin khi giao dịch"**:
- `ppa`, `spi` (bắt buộc)
- `edi` (khát khao và động lực)
- `mpi` (hình ảnh bên ngoài)
- `cmi` (kiểm soát cảm xúc)
- `nei` (lợi thế bẩm sinh)

#### **"Tôi muốn biết về tính cách của mình"**:
- `ppa`, `spi` (bắt buộc)
- `mpi` (tính cách chính)
- `edi` (bản chất bên trong)
- `nei` (đặc điểm bẩm sinh)
- `sai` (sở thích và thế mạnh)

#### **"Tôi nên tập trung vào chiến lược nào?"**:
- `ppa`, `spi` (bắt buộc)
- `ri` (chiến lược đỉnh phong độ)
- `ari` (tư duy phân tích)
- `cii` (phù hợp với xu hướng)

## YÊU CẦU:
- Chọn 5-6 chỉ số phù hợp nhất (không quá nhiều)
- Nếu câu hỏi đề cập đến thời gian thì ưu tiên chọn những chỉ số về thời gian
- Ưu tiên chỉ số có thể giải quyết vấn đề cụ thể
- Đảm bảo coverage toàn diện (tâm lý + hiệu suất + timing + phát triển)
- Trả về JSON chính xác theo format

{format_instructions}"""
        ),
        ("human", "Câu hỏi của người dùng: {question}"),
    ])
    
    # Processing string
    key_selection_chain = (
        {
            "question": RunnablePassthrough(), 
            "format_instructions": RunnableLambda(lambda x: parser.get_format_instructions()),
            "indicator_definitions": RunnableLambda(lambda x: indicator_definitions)
        }
        | prompt
        | llm
        | parser
    )

    try:
        # Call chain and get result
        result = key_selection_chain.invoke({"question": question})
        selected_keys = result.get("selected_keys", [])
        
        # Ensure the priority indicators always present
        final_keys = list(dict.fromkeys(["ppa", "spi"] + selected_keys))
        
        # Limit the number of indicators to avoid overloading
        if len(final_keys) > 6:
            # Priority indicators
            priority_indicators = ["ppa", "spi"]
            final_keys = [key for key in final_keys if key in priority_indicators or len([k for k in final_keys if k in priority_indicators]) < 6]
            final_keys = final_keys[:6]
        
        print(f"🔍 LLM selected TBI indicators: {final_keys}")
        return final_keys
        
    except Exception as e:
        print(f"Error when calling LLM to select TBI indicators: {e}")
        # Fallback to basic indicators
        return ["ppa", "spi", "edi", "cmi"]


def _calculate_tbi_insights(birthday: str, age_tci: List[int], current_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate TBI insights from indicators.
    
    Args:
        birthday: Birthday (dd/mm/yyyy)
        age_tci: List of age TCI
        current_date: Current date (dd/mm/yyyy), default is today   
    
    Returns:
        Dict containing TBI insights and analysis
    """
    from datetime import datetime
    import pytz
    
    # Parse dates
    try:
        dob = datetime.strptime(birthday, '%d/%m/%Y')
        if current_date:
            current = datetime.strptime(current_date, '%d/%m/%Y')
        else:
            # Use current time in Vietnam timezone
            vntz = pytz.timezone("Asia/Ho_Chi_Minh")
            current = datetime.now(vntz)
        
        # Calculate current age
        age = current.year - dob.year
        if current.month < dob.month or (current.month == dob.month and current.day < dob.day):
            age -= 1
        
        # Determine current TCI phase based on personal age_tci
        current_tci = 1  # Default to first phase
        tci_name = "tci_1"
        
        for i, tci_age in enumerate(age_tci, 1):
            if age >= tci_age:
                current_tci = i
                tci_name = f"tci_{i}"
            else:
                break
        
        # Get corresponding BCI based on TCI
        bci_name = f"bci_{current_tci}"
        
        # Get TCI age info
        current_tci_age = age_tci[current_tci - 1] if current_tci <= len(age_tci) else age_tci[-1]
        next_tci_age = age_tci[current_tci] if current_tci < len(age_tci) else None
        
        return {
            "current_age": age,
            "current_tci": current_tci,
            "tci_name": tci_name,
            "bci_name": bci_name,
            "current_tci_age": current_tci_age,
            "next_tci_age": next_tci_age,
            "milestone_description": f"Bạn đang ở giai đoạn {current_tci} (tuổi {age}/{current_tci_age})",
            "challenge_description": f"Thách thức tương ứng với giai đoạn {current_tci}",
            "age_tci": age_tci
        }
        
    except Exception as e:
        print(f"Error calculating TBI insights: {e}")
        return {
            "current_age": None,
            "current_tci": 1,
            "tci_name": "tci_1",
            "bci_name": "bci_1",
            "current_tci_age": age_tci[0] if age_tci else None,
            "next_tci_age": age_tci[1] if len(age_tci) > 1 else None,
            "milestone_description": "Không thể xác định giai đoạn hiện tại",
            "challenge_description": "Không thể xác định thách thức hiện tại",
            "age_tci": age_tci
        }

def _calculate_tci_insights(birthday: str, age_tci: List[int], current_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate additional insights based on TCI indicators.
    
    Args:
        birthday: Birthday (dd/mm/yyyy)
        age_tci: List of age TCI
        current_date: Current date (dd/mm/yyyy), default is today   
    
    Returns:
        Dict containing additional insights and analysis
    """
    from datetime import datetime
    import pytz
    
    # Parse dates
    try:
        dob = datetime.strptime(birthday, '%d/%m/%Y')
        if current_date:
            current = datetime.strptime(current_date, '%d/%m/%Y')
        else:
            # Use current time in Vietnam timezone
            vntz = pytz.timezone("Asia/Ho_Chi_Minh")
            current = datetime.now(vntz)
        
        # Calculate current age
        age = current.year - dob.year
        if current.month < dob.month or (current.month == dob.month and current.day < dob.day):
            age -= 1
        
        # Determine current milestone based on personal age_milestones
        current_tci = 1  # Default to first milestone
        tci_name = "tci_1"
        
        for i, tci_age in enumerate(age_tci, 1):
            if age >= tci_age:
                current_tci = i
                tci_name = f"tci_{i}"
            else:
                break
        
        # Get corresponding challenge
        bci_name = f"bci_{current_tci}"
        
        # Get milestone age info
        current_tci_age = age_tci[current_tci - 1] if current_tci <= len(age_tci) else age_tci[-1]
        next_tci_age = age_tci[current_tci] if current_tci < len(age_tci) else None
        
        return {
            "current_age": age,
            "current_tci": current_tci,
            "tci_name": tci_name,
            "bci_name": bci_name,
            "current_tci_age": current_tci_age,
            "next_tci_age": next_tci_age,
            "milestone_description": f"Bạn đang ở giai đoạn {current_tci} (tuổi {age}/{current_tci_age})",
            "challenge_description": f"Thách thức tương ứng với giai đoạn {current_tci}",
            "age_tci": age_tci
        }
        
    except Exception as e:
        print(f"Error calculating milestone: {e}")
        return {
            "current_age": None,
            "current_tci": 1,
            "tci_name": "tci_1",
            "bci_name": "bci_1",
            "current_tci_age": age_tci[0] if age_tci else None,
            "next_tci_age": age_tci[1] if len(age_tci) > 1 else None,
            "tci_description": "Không thể xác định giai đoạn hiện tại",
            "bci_description": "Không thể xác định thách thức hiện tại",
            "age_tci": age_tci
        }

def _prepare_data(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    question: str = input_dict["question"]
    user_name: Optional[str] = input_dict.get("user_name")
    birthday: Optional[str] = input_dict.get("birthday")
    current_day: Optional[str] = input_dict.get("current_day")
    language: str = input_dict.get("language", "vi")  # Extract language parameter
    
    # Use manual input if provided, otherwise fallback to default
    if user_name and birthday:
        profile = {"dob": birthday, "name": user_name}
    else:
        # Default fallback profile - you should implement actual parsing logic here
        # For now, using placeholder values
        profile = {"dob": "01/01/1990", "name": "Default User"}

    # Validate current_day: accept dd/mm/yyyy; if invalid/empty/None, let CalNum default to VN time
    def _normalize_current_day(day_str: Optional[str]) -> Optional[str]:
        if not day_str:
            return None
        s = str(day_str).strip()
        if not s:
            return None
        import re
        if re.fullmatch(r"\d{2}/\d{2}/\d{4}", s):
            return s
        return None

    normalized_current = _normalize_current_day(current_day)

    # Handle None case for TBICalculator - create TBICalculator based on whether current_date is available
    if normalized_current is not None:
        cal = TBICalculator(dob=profile["dob"], name=profile["name"], current_date=normalized_current)
    else:
        # Let TBICalculator handle default current time by passing empty string or using a dummy date
        from datetime import datetime
        import pytz
        vntz = pytz.timezone("Asia/Ho_Chi_Minh")
        current_vn = datetime.now(vntz).strftime("%d/%m/%Y")
        cal = TBICalculator(dob=profile["dob"], name=profile["name"], current_date=current_vn)
    numbers = cal.get_all_tbi_indicators()

    # Use LLM to select indicators
    selected_keys = _infer_keys_from_llm(question)

    # Get age_milestones from CalNum calculation - ensure it's a list
    age_tci_raw = numbers.get("age_tci", [])
    age_tci = age_tci_raw if isinstance(age_tci_raw, list) else []
    
    # Calculate current milestone and challenge based on user's personal age_milestones
    milestone_info = _calculate_tci_insights(profile["dob"], age_tci, normalized_current)
    
    # If milestone or challenge indicators are selected, prioritize current ones
    tci_indicators = [k for k in selected_keys if k.startswith("tci_")]
    bci_indicators = [k for k in selected_keys if k.startswith("bci_")]
    
    if tci_indicators or bci_indicators:
        # Replace generic milestone/challenge with current ones
        for i, key in enumerate(selected_keys):
            if key.startswith("tci_"):
                selected_keys[i] = milestone_info["tci_name"]
            elif key.startswith("bci_"):
                selected_keys[i] = milestone_info["bci_name"]
        
        # Remove duplicates after replacement
        selected_keys = list(dict.fromkeys(selected_keys))

    # Fetch S3 docs for all selected indicators
    s3 = S3Client()
    docs: Dict[str, str] = {}

    # Map indicator keys to their corresponding number values and S3 types
    # Safe get with type checking
    def safe_nested_get(data: Any, *keys: str) -> Any:
        """Safely get nested dictionary values"""
        result = data
        for key in keys:
            if isinstance(result, dict):
                result = result.get(key)
            else:
                return None
        return result
    
    indicator_mapping = {
        "ppa": ("ppa", numbers.get("ppa")),
        "spi": ("spi", numbers.get("spi")),
        "edi": ("edi", numbers.get("edi")),
        "mpi": ("mpi", numbers.get("mpi")),
        "cmi": ("cmi", numbers.get("cmi")),
        "ri": ("ri", numbers.get("ri")),
        "sai": ("sai", numbers.get("sai")),
        "ppai": ("ppai", numbers.get("ppai")),
        "ioci": ("ioci", numbers.get("ioci")),
        "bci_1": ("bci_1", safe_nested_get(numbers, "bci", "bci_1")),
        "bci_2": ("bci_2", safe_nested_get(numbers, "bci", "bci_2")),
        "bci_3": ("bci_3", safe_nested_get(numbers, "bci", "bci_3")),
        "bci_4": ("bci_4", safe_nested_get(numbers, "bci", "bci_4")),
        "tci_1": ("tci_1", safe_nested_get(numbers, "tci_phase", "tci_1")),
        "tci_2": ("tci_2", safe_nested_get(numbers, "tci_phase", "tci_2")),
        "tci_3": ("tci_3", safe_nested_get(numbers, "tci_phase", "tci_3")),
        "tci_4": ("tci_4", safe_nested_get(numbers, "tci_phase", "tci_4")),
        "ari": ("ari", numbers.get("ari")),
        "dai": ("dai", safe_nested_get(numbers, "alignment_signals", "dai")),
        "ami": ("ami", safe_nested_get(numbers, "alignment_signals", "ami")),
        "mri": ("mri", safe_nested_get(numbers, "alignment_signals", "mri")),
        "nei": ("nei", numbers.get("nei")),
        "ssi": ("ssi", numbers.get("ssi")),
        "wmi": ("wmi", numbers.get("wmi")),
        "tai": ("tai", numbers.get("tai")),
        "cii": ("cii", numbers.get("cii")),
        "bli": ("bli", numbers.get("bli")),
    }

    # Prefetch S3 docs for current milestone/challenge so they are always available
    try:
        current_tci_val = milestone_info.get("current_tci")
        if current_tci_val is not None:
            current_tci_ord = int(current_tci_val)
            tci_key = milestone_info.get("tci_name")
            if tci_key:
                tci_phase_data = numbers.get("tci_phase", {})
                if isinstance(tci_phase_data, dict):
                    tci_val = tci_phase_data.get(tci_key)
                    if isinstance(tci_val, int):
                        try:
                            doc_content = s3.get_document_text_for_numerology(
                                "tci", tci_val, tci_number=current_tci_ord
                            )
                            if doc_content and not doc_content.startswith("Lỗi"):
                                docs[tci_key] = doc_content
                        except Exception:
                            pass
            bci_key = milestone_info.get("bci_name")
            if bci_key:
                bci_data = numbers.get("bci", {})
                if isinstance(bci_data, dict):
                    bci_val = bci_data.get(bci_key)
                    if isinstance(bci_val, int):
                        try:
                            doc_content = s3.get_document_text_for_numerology(
                                "bci", bci_val, bci_number=current_tci_ord
                            )
                            if doc_content and not doc_content.startswith("Lỗi"):
                                docs[bci_key] = doc_content
                        except Exception:
                            pass
    except Exception:
        pass

    # Fetch documents for all selected keys
    print(f"Fetching documents for {len(selected_keys)} selected keys...")
    print(f"Current tci info: {milestone_info}")
    
    for key in selected_keys:
        print(f"  Processing key: {key}")
        
        # Special handling: milestone_X and challenge_X should fetch S3 docs by ordinal (1..4)
        if key.startswith("tci_"):
            try:
                tci_ord = int(key.split("_")[1])
                # File name expects milestone value (calculated), folder expects ordinal 1..4
                tci_phase_data = numbers.get("tci_phase", {})
                if isinstance(tci_phase_data, dict):
                    tci_value = tci_phase_data.get(f"tci_{tci_ord}")
                    if not isinstance(tci_value, int):
                        docs[f"{key}_error"] = f"Invalid tci value: {tci_value}"
                        print(f"Invalid tci value: {tci_value}")
                        continue
                    try:
                        doc_content = s3.get_document_text_for_numerology(
                            "tci",
                            tci_value,
                            tci_number=tci_ord,
                        )
                        if doc_content:
                            docs[key] = doc_content
                            print(f"Tci doc fetched: {len(str(doc_content))} chars")
                        else:
                            docs[f"{key}_error"] = "Empty content"
                            print(f"Tci doc fetch returned empty content")
                    except Exception as e:
                        docs[f"{key}_error"] = str(e)
                        print(f"Tci doc exception: {e}")
                    continue
                else:
                    docs[f"{key}_error"] = f"Invalid tci_phase data type: {type(tci_phase_data)}"
                    print(f"Invalid tci_phase data type: {type(tci_phase_data)}")
                    continue
            except Exception as e:
                print(f"Invalid tci key '{key}': {e}")

        if key.startswith("bci_"):
            try:
                bci_ord = int(key.split("_")[1])
                # File name expects challenge value (calculated), folder expects ordinal 1..4
                bci_data = numbers.get("bci", {})
                if isinstance(bci_data, dict):
                    bci_value = bci_data.get(f"bci_{bci_ord}")
                    if not isinstance(bci_value, int):
                        docs[f"{key}_error"] = f"Invalid bci value: {bci_value}"
                        print(f"Invalid bci value: {bci_value}")
                        continue
                    try:
                        doc_content = s3.get_document_text_for_numerology(
                            "bci",
                            bci_value,
                            bci_number=bci_ord,
                        )
                        if doc_content:
                            docs[key] = doc_content
                            print(f"Bci doc fetched: {len(str(doc_content))} chars")
                        else:
                            docs[f"{key}_error"] = "Empty content"
                            print(f"Bci doc fetch returned empty content")
                    except Exception as e:
                        docs[f"{key}_error"] = str(e)
                        print(f"Bci doc exception: {e}")
                    continue
                else:
                    docs[f"{key}_error"] = f"Invalid bci data type: {type(bci_data)}"
                    print(f"Invalid bci data type: {type(bci_data)}")
                    continue
            except Exception as e:
                print(f"Invalid bci key '{key}': {e}")

        if key in indicator_mapping:
            s3_type, number_value = indicator_mapping[key]

            if isinstance(number_value, int):
                try:
                    doc_content = s3.get_document_text_for_numerology(s3_type, number_value)
                    if doc_content and not doc_content.startswith("Error"):
                        docs[key] = doc_content
                        print(f"Document fetched: {len(doc_content)} chars")
                    else:
                        docs[f"{key}_error"] = doc_content
                        print(f"Document fetch failed: {doc_content}")
                except Exception as e:
                    docs[f"{key}_error"] = str(e)
                    print(f"Exception: {e}")
            # Handle list values (for passion and missing_aspects)
            elif isinstance(number_value, list) and len(number_value) > 0:
                print(f"Processing list of {len(number_value)} values: {number_value}")
                combined_content = []
                for i, num in enumerate(number_value):
                    if isinstance(num, int):
                        try:
                            doc_content = s3.get_document_text_for_numerology(s3_type, num)
                            if doc_content and not doc_content.startswith("Error"):
                                combined_content.append(f"--- Số {num} ---\n{doc_content}")
                                # print(f"Document {i+1} fetched for number {num}: {len(doc_content)} chars")
                            else:
                                # print(f"Document {i+1} fetch failed for number {num}: {doc_content}")
                                pass
                        except Exception as e:
                            print(f"Exception for number {num}: {e}")
                    else:
                        print(f"Invalid number in list: {num}")
                
                if combined_content:
                    docs[key] = "\n\n".join(combined_content)
                    print(f"Combined documents fetched: {len(docs[key])} chars total")
                else:
                    docs[f"{key}_error"] = "No valid documents could be fetched from the list"
                    print(f"No valid documents fetched from list")
            else:
                docs[f"{key}_error"] = f"Invalid number value: {number_value}"
                print(f"Invalid number value: {number_value}")
        else:
            # For indicators not in S3 mapping, use the calculated values and meanings
            print(f"    Using calculated value for: {key}")
            
            if key in numbers:
                if key not in docs:
                    docs[key] = f"Value: {numbers[key]}"
            elif key == milestone_info["tci_name"]:
                # Current milestone with age context
                if key not in docs:
                    tci_phase_data = numbers.get("tci_phase", {})
                    if isinstance(tci_phase_data, dict):
                        tci_value = tci_phase_data.get(f"tci_{milestone_info['current_tci']}")
                        docs[key] = f"{milestone_info.get('tci_description', 'N/A')} - Value: {tci_value}"
            elif key == milestone_info["bci_name"]:
                # Current challenge with age context
                if key not in docs:
                    bci_data = numbers.get("bci", {})
                    if isinstance(bci_data, dict):
                        bci_value = bci_data.get(f"bci_{milestone_info.get('current_bci', 1)}")
                        docs[key] = f"{milestone_info.get('bci_description', 'N/A')} - Value: {bci_value}"
            elif key.startswith("bci_"):
                bci_num = key.split("_")[1]
                bci_data = numbers.get("bci", {})
                if isinstance(bci_data, dict):
                    bci_value = bci_data.get(f"bci_{bci_num}")
                    if bci_value is not None and key not in docs:
                        docs[key] = f"Thách thức {bci_num}: {bci_value}"
            elif key.startswith("tci_"):
                tci_num = key.split("_")[1]
                tci_phase_data = numbers.get("tci_phase", {})
                if isinstance(tci_phase_data, dict):
                    tci_value = tci_phase_data.get(f"tci_{tci_num}")
                    if tci_value is not None and key not in docs:
                        docs[key] = f"Giai đoạn {tci_num}: {tci_value}"
            else:
                if key not in docs:
                    docs[key] = f"Value: {numbers.get(key, 'N/A')}"

    # Provide mapping meanings for selected keys
    meanings: Dict[str, str] = {}
    for k in selected_keys:
        if k in tbi_definitions:
            meanings[k] = tbi_definitions[k]

    # Calculate TBI insights
    insights = _calculate_tbi_insights(profile["dob"], age_tci, normalized_current)
    
    # Prepare user info
    user_info = f"Tên: {profile['name']}, Ngày sinh: {profile['dob']}"
    
    # Prepare TBI indicators summary
    tbi_indicators = {}
    for key in selected_keys:
        if key in numbers:
            tbi_indicators[key] = numbers[key]
        elif key in docs:
            tbi_indicators[key] = f"Document available ({len(docs[key])} chars)"
    
    # Prepare analysis context
    analysis_context = f"Giai đoạn hiện tại: {milestone_info.get('tci_description', 'N/A')}"
    
    # Return both structured payload and text resources
    return {
        "question": question,
        "user_info": user_info,
        "selected_keys": selected_keys,
        "tbi_indicators": tbi_indicators,
        "meanings": meanings,
        "documents": docs,
        "insights": insights,
        "analysis_context": analysis_context,
        "language": language,  # Include language in return value
    }


def build_tbi_agent():
    """
    Build TBI (Trading Behavior Intelligence) agent with Jinja2 template support
    
    Returns:
        Chain for TBI analysis
    """
    prepare = RunnableLambda(_prepare_data)

    def _render_prompt_with_jinja(data: Dict[str, Any]) -> str:
        """Render the Jinja2 template with provided data"""
        template_content = _read_prompt()
        
        # Create Jinja2 template
        template = Template(template_content)
        
        # Extract TBI context from data
        selected_keys = data.get("selected_keys", [])
        meanings = data.get("meanings", {})
        # insights = data.get("insights", {})
        documents = data.get("documents", {})
        
        # Prepare TBI context - combining meanings and correspond documents for selected keys
        tbi_context_parts = []
        for key in selected_keys:
            if key in meanings:
                context_part = f"**{key.upper()}**: {meanings[key]}"
                # Add documents if available
                if key in documents:
                    context_part += f"\n- Tài liệu: {documents[key]}"
                tbi_context_parts.append(context_part)
        
        tbi_context = "\n\n".join(tbi_context_parts)
        
        # Render template with data
        rendered_content = template.render(
            tbi_context=tbi_context,
            tbi_question=data.get("question", ""),
            language=data.get("language", "vi")
        )
        
        return rendered_content

    def _create_prompt_with_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Create prompt with rendered Jinja2 template"""
        rendered_prompt = _render_prompt_with_jinja(data)
        
        # Create the system message with rendered template
        return {
            "system_message": rendered_prompt,
            "human_message": f"Câu hỏi: {data.get('question', '')}"
        }

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_message}"),
        ("human", "{human_message}"),
    ])

    llm = get_openai_llm()
    
    # Build the chain: prepare_data -> render_template -> prompt -> llm
    chain = (
        prepare 
        | RunnableLambda(_create_prompt_with_data)
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain
