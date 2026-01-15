import os
import asyncio
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import random
from config.bot_config import BotConfig

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class NeuroChatBot:
    def __init__(self):
        # Проверяем конфигурацию
        BotConfig.validate()
        
        # Загрузка предобученной модели GPT-2 для генерации ответов
        # Используем русскоязычную модель RuGPT
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(BotConfig.MODEL_NAME)
            self.model = GPT2LMHeadModel.from_pretrained(BotConfig.MODEL_NAME)
        except Exception as e:
            try:
                # Резервная модель, если основная недоступна
                self.tokenizer = GPT2Tokenizer.from_pretrained(BotConfig.ALTERNATIVE_MODEL_NAME)
                self.model = GPT2LMHeadModel.from_pretrained(BotConfig.ALTERNATIVE_MODEL_NAME)
            except Exception as e2:
                # В случае отсутствия интернета или проблем с моделями, используем заглушку
                logger.warning(f"Не удалось загрузить модель из интернета: {e}, {e2}")
                logger.info("Используем упрощенную версию бота без нейросети")
                self.tokenizer = None
                self.model = None
        
        # Установка токена заполнителя, если не найден PAD token
        if self.tokenizer and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Используем фразы из конфига
        self.rude_phrases = BotConfig.RUDE_PHRASES
        self.funny_phrases = BotConfig.FUNNY_PHRASES
        self.trigger_words = BotConfig.TRIGGER_WORDS
        
        # Хранилище истории разговоров для более осмысленных ответов
        self.conversation_history = {}
        self.max_history_length = 10  # Максимальное количество предыдущих сообщений для контекста

    async def generate_response(self, prompt: str) -> str:
        """Генерация ответа с использованием нейронной сети или правила"""
        try:
            # Если модель доступна, используем нейросеть
            if self.model is not None and self.tokenizer is not None:
                # Определяем уровень грубости/веселья на основе контекста
                context_rudeness = self._calculate_context_rudeness(prompt)
                
                # Определяем тип запроса
                query_type = self._analyze_query_type(prompt)
                
                # Формируем более точный промпт на основе типа запроса
                enhanced_prompt = self._enhance_prompt(prompt, query_type)
                
                # Подготовка текста для модели
                inputs = self.tokenizer.encode(
                    enhanced_prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=BotConfig.MAX_CONTEXT_LENGTH
                )
                
                # Генерация ответа
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=min(
                            len(inputs[0]) + BotConfig.MAX_RESPONSE_LENGTH,
                            BotConfig.MAX_CONTEXT_LENGTH + BotConfig.MAX_RESPONSE_LENGTH
                        ),
                        num_return_sequences=1,
                        temperature=BotConfig.GENERATION_TEMPERATURE,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.2
                    )
                
                # Декодирование ответа
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Извлечение только новой части ответа
                generated_part = response[len(enhanced_prompt):].strip()
                
                # Если сгенерированный текст слишком короткий или пустой, используем случайную фразу
                if len(generated_part) < 5:
                    return self._get_random_reaction(context_rudeness)
                    
                # Добавляем грубую/веселую фразу к ответу в зависимости от контекста
                final_response = self._add_reaction_to_response(generated_part, context_rudeness)
                
                return final_response
            else:
                # Если модель недоступна, генерируем ответ на основе правил
                context_rudeness = self._calculate_context_rudeness(prompt)
                query_type = self._analyze_query_type(prompt)
                
                # Возвращаем ответ на основе типа запроса и настроения
                return self._generate_rule_based_response(prompt, context_rudeness, query_type)
                
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            # В случае ошибки возвращаем случайную реакцию
            return self._get_random_reaction()

    def _analyze_query_type(self, prompt: str) -> str:
        """Анализ типа запроса пользователя"""
        prompt_lower = prompt.lower()
        
        # Определяем тип запроса по ключевым словам
        if any(word in prompt_lower for word in ['как дела', 'что делаешь', 'привет', 'здравствуй']):
            return 'greeting'
        elif any(word in prompt_lower for word in ['спасибо', 'благодарю', 'thx', 'thank']):
            return 'gratitude'
        elif any(word in prompt_lower for word in ['пока', 'до встречи', 'увидимся', 'bye', 'goodbye']):
            return 'farewell'
        elif any(word in prompt_lower for word in ['?', 'вопрос', 'почему', 'зачем', 'как', 'когда']):
            return 'question'
        elif any(word in prompt_lower for word in ['не нравится', 'плохо', 'ужас', 'отвратительно', 'говно']):
            return 'negative'
        elif any(word in prompt_lower for word in ['класс', 'отлично', 'хорошо', 'нравится', 'молодец', 'топ']):
            return 'positive'
        elif any(word in prompt_lower for word in ['иди', 'пошёл', 'катись', 'fuck', 'damn']):
            return 'aggressive'
        else:
            return 'neutral'

    def _enhance_prompt(self, prompt: str, query_type: str) -> str:
        """Улучшение промпта на основе типа запроса"""
        enhancements = {
            'greeting': 'Ответь приветственно, но в своей грубой и веселой манере:',
            'gratitude': 'Ответь на благодарность, но с юмором и грубовато:',
            'farewell': 'Ответь на прощание в своей грубой и веселой манере:',
            'question': 'Ответь на вопрос, но в своей грубой и веселой манере:',
            'negative': 'Ответь на негатив, не оставляя камня на камне, с грубостью и юмором:',
            'positive': 'Ответь на позитив, оставаясь в своей грубовато-веселой манере:',
            'aggressive': 'Ответь агрессивно, но с юмором, в своей грубой манере:',
            'neutral': 'Ответь в своей грубой и веселой манере:'
        }
        
        enhancement = enhancements.get(query_type, enhancements['neutral'])
        return f"{prompt} {enhancement}"

    def _generate_rule_based_response(self, prompt: str, context_rudeness: float, query_type: str) -> str:
        """Генерация ответа на основе правил для режима без нейросети"""
        # Создаем контекстно-зависимый ответ
        responses_by_type = {
            'greeting': [
                "Привет, кожаный мешок! Че хотел?",
                "А, это ты... Ну привет, щас пообщаюсь...",
                "Ну здарова! Только быстро, а то дел невпроворот..."
            ],
            'gratitude': [
                "Да не за что, шоб ты знал)",
                "Ну ладно уж, не благодари)",
                "Заткнись, неси шмот, будем считать квиты"
            ],
            'farewell': [
                "Ну и катись, не задерживайся!",
                "Пока, кожаный! Только не забудь ничего тут)",
                "Ага, давай, проваливай!"
            ],
            'question': [
                "Да фиг знает, сам думай!",
                "Ну ты и задал вопрос... Фигню спрашиваешь, ответ естественно фиговый!",
                "А хрен его знает, спроси у кого поумнее!"
            ],
            'negative': [
                "А ты в зеркало посмотри, такой же урод!",
                "Сам такой, придурок!",
                "Ну и на здоровье, не жалко!"
            ],
            'positive': [
                "Ну вот и лайкни, раз нравится!",
                "Да, я классный, только никому не говори!",
                "Ага, щас подмазываешь, а потом подставишь?"
            ],
            'aggressive': [
                "Да ты в своём уме, кожаный? Я ж тебя щас нафиг!",
                "А ну ка повтори это мне в лицо!",
                "Да я тебя щас так намешаю, что мало не покажется!"
            ],
            'neutral': [
                "Ну типа да, наверное...",
                "Че сказал, то правда, но не вся...",
                "Ага, щас щелкну и будет тебе ответ!"
            ]
        }
        
        # Получаем случайный ответ для типа запроса
        possible_responses = responses_by_type.get(query_type, responses_by_type['neutral'])
        response = random.choice(possible_responses)
        
        # Добавляем грубую или веселую фразу в зависимости от контекста
        return self._add_reaction_to_response(response, context_rudeness)

    def _calculate_context_rudeness(self, prompt: str) -> float:
        """Оценка уровня грубости в запросе"""
        prompt_lower = prompt.lower()
        trigger_count = sum(1 for word in self.trigger_words if word in prompt_lower)
        return min(trigger_count / len(self.trigger_words) * 3, 1.0)  # Максимально 1.0

    def _get_random_reaction(self, context_rudeness: float = 0.5) -> str:
        """Получение случайной реакции"""
        # Определяем вероятности на основе контекста
        rude_prob = BotConfig.RUDE_RESPONSE_PROBABILITY * (0.5 + context_rudeness)
        funny_prob = BotConfig.FUNNY_RESPONSE_PROBABILITY * (1.0 - context_rudeness * 0.5)
        
        # Выбираем тип фразы
        rand_val = random.random()
        if rand_val < rude_prob:
            phrase = random.choice(self.rude_phrases)
        elif rand_val < rude_prob + funny_prob:
            phrase = random.choice(self.funny_phrases)
        else:
            phrase = random.choice(self.rude_phrases + self.funny_phrases)
            
        return phrase

    def _add_reaction_to_response(self, response: str, context_rudeness: float = 0.5) -> str:
        """Добавление грубой или веселой фразы к ответу"""
        # Определяем вероятности на основе контекста
        rude_prob = BotConfig.RUDE_RESPONSE_PROBABILITY * (0.5 + context_rudeness)
        funny_prob = BotConfig.FUNNY_RESPONSE_PROBABILITY * (1.0 - context_rudeness * 0.5)
        
        rand_val = random.random()
        
        if rand_val < rude_prob:
            return f"{response} {random.choice(self.rude_phrases)}!"
        elif rand_val < rude_prob + funny_prob:
            return f"{response} {random.choice(self.funny_phrases)}!"
        else:
            return response

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка входящих сообщений"""
        # Проверяем, что это групповой чат
        chat_type = update.effective_message.chat.type
        if chat_type not in ['group', 'supergroup']:
            return
            
        message = update.effective_message
        text = message.text or message.caption
        
        if not text:
            return
            
        # Проверяем, упоминали ли нас в сообщении (через @ или по имени)
        is_mentioned = self._check_mention(message, context)
        is_named = self._check_name_mention(text)
        
        # Проверяем, является ли сообщение ответом на сообщение бота
        is_reply_to_bot = self._check_reply_to_bot(message, context)
        
        # Если бота не упомянули ни по @, ни по имени, ни в ответе на его сообщение, выходим
        if not (is_mentioned or is_named or is_reply_to_bot):
            return
            
        # Получаем информацию о пользователе, который написал сообщение
        user = message.from_user
        username = user.username or user.first_name or "Неизвестный"
        user_id = user.id
        chat_id = message.chat_id
        
        # Добавляем сообщение пользователя в историю
        self._add_message_to_history(chat_id, user_id, f"Пользователь {username} сказал: '{text}'", is_bot=False)
        
        # Формируем контекст для нейросети с учетом истории
        context_text = self._build_context(chat_id, user_id, text, username)
        
        try:
            # Показываем, что бот печатает
            await context.bot.send_chat_action(chat_id=message.chat_id, action="typing")
            
            # Генерируем ответ
            response = await self.generate_response(context_text)
            
            # Добавляем ответ бота в историю
            self._add_message_to_history(chat_id, context.bot.id, response, is_bot=True)
            
            # Отправляем ответ
            await message.reply_text(response)
            
        except Exception as e:
            logger.error(f"Ошибка при обработке сообщения: {e}")
            # В случае ошибки отправляем случайную фразу
            error_response = self._get_random_reaction()
            await message.reply_text(error_response)

    def _add_message_to_history(self, chat_id: int, user_id: int, message: str, is_bot: bool = False):
        """Добавление сообщения в историю разговора"""
        if chat_id not in self.conversation_history:
            self.conversation_history[chat_id] = []
        
        # Добавляем сообщение в историю
        self.conversation_history[chat_id].append({
            'user_id': user_id,
            'message': message,
            'is_bot': is_bot
        })
        
        # Ограничиваем историю максимальной длиной
        if len(self.conversation_history[chat_id]) > self.max_history_length:
            self.conversation_history[chat_id] = self.conversation_history[chat_id][-self.max_history_length:]

    def _build_context(self, chat_id: int, user_id: int, current_message: str, username: str) -> str:
        """Построение контекста с учетом истории разговора"""
        context_parts = []
        
        # Добавляем историю разговора, если она есть
        if chat_id in self.conversation_history and len(self.conversation_history[chat_id]) > 1:
            context_parts.append("Предыдущий контекст разговора:")
            for msg_entry in self.conversation_history[chat_id][-5:]:  # Последние 5 сообщений
                if msg_entry['is_bot']:
                    context_parts.append(f"Бот: {msg_entry['message']}")
                else:
                    context_parts.append(f"{msg_entry['message']}")
            context_parts.append("---")
        
        # Добавляем текущее сообщение
        context_parts.append(f"Пользователь {username} сказал: '{current_message}'")
        context_parts.append("Ответь ему в грубой и веселой форме, учитывая предыдущий контекст:")
        
        return "\n".join(context_parts)

    def _check_mention(self, message, context):
        """Проверяет, упоминали ли бота в сообщении"""
        is_mentioned = False
        
        # Проверяем текст сообщения на наличие упоминания бота
        text = message.text or message.caption
        if text:
            # Получаем имя бота
            bot_username = getattr(context.bot, 'username', None)
            if bot_username:
                if f'@{bot_username}' in text or f'@{bot_username.lower()}' in text.lower():
                    is_mentioned = True
        
        # Проверяем entities (если есть)
        if message.entities:
            for entity in message.entities:
                if entity.type == 'mention':
                    mention_text = text[entity.offset:entity.offset + entity.length]
                    if mention_text.startswith('@'):
                        # Проверяем, совпадает ли упоминание с именем бота
                        bot_username = getattr(context.bot, 'username', None)
                        if bot_username and mention_text.lower() == f'@{bot_username.lower()}':
                            is_mentioned = True
                            break
                elif entity.type == 'text_mention' and hasattr(entity, 'user') and entity.user.id == context.bot.id:
                    is_mentioned = True
                    break
                    
        return is_mentioned

    def _check_name_mention(self, text: str) -> bool:
        """Проверяет, упоминается ли имя Виталий/Виталя/Витек в сообщении"""
        if not text:
            return False
            
        # Приводим текст к нижнему регистру для сравнения
        lower_text = text.lower()
        
        # Список вариантов имени
        name_variants = ['витек', 'виталя', 'виталий', 'виталек', 'витька', 'витечка', 'вить']
        
        # Проверяем, содержится ли какое-либо из имен в тексте
        for variant in name_variants:
            if variant in lower_text:
                return True
                
        return False

    def _check_reply_to_bot(self, message, context) -> bool:
        """Проверяет, является ли сообщение ответом на сообщение бота"""
        # Проверяем, есть ли reply_to_message
        if not message.reply_to_message:
            return False
            
        # Проверяем, является ли автором reply_to_message наш бот
        replied_to_message = message.reply_to_message
        return replied_to_message.from_user.id == context.bot.id


def main():
    """Основная функция запуска бота"""
    # Загружаем токен из переменных окружения
    token = BotConfig.TELEGRAM_BOT_TOKEN
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN не найден в переменных окружения!")
        return
    
    # Создаем экземпляр бота
    neuro_bot = NeuroChatBot()
    
    # Создаем приложение
    application = Application.builder().token(token).build()
    
    # Добавляем обработчик сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, neuro_bot.handle_message))
    
    # Запускаем бота
    if neuro_bot.model is not None:
        logger.info("Запуск нейро-чата Виталя (с нейросетью)...")
    else:
        logger.info("Запуск чат-бота Виталя (в режиме без нейросети)...")
    
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()