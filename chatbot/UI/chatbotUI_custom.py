user_style = """
<div style="display: flex; justify-content: flex-end; align-items: center; margin: 8px 0;">
    <div style="background-color: #DCF8C6; color: #000000; border-radius: 12px; 
                padding: 8px 16px; max-width: 60%; width: fit-content; height: fit-content; 
                box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1); word-wrap: break-word;">
        {content}
    </div>
    <div style="width: 32px; height: 32px; border-radius: 50%; 
                background-color: #87CEEB; display: flex; justify-content: center; 
                align-items: center; margin-left: 8px;">
        <span style="color: white; font-weight: bold;">C</span>
    </div>
</div>
"""

bot_style = """
<div style="display: flex; justify-content: flex-start; align-items: center; margin: 8px 0;">
    <div style="width: 32px; height: 32px; border-radius: 50%; 
                background-color: #F39C12; display: flex; justify-content: center; 
                align-items: center; margin-right: 8px;">
        <span style="color: white; font-weight: bold;">A</span>
    </div>
    <div style="background-color: #FFFFFF; color: #000000; border: 1px solid #E6E6E6; 
                border-radius: 12px; padding: 8px 16px; max-width: 60%; 
                width: fit-content; height: fit-content; 
                box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1); word-wrap: break-word;">
        {content}
    </div>
</div>
"""