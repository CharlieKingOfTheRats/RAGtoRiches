import { useState, useRef, useEffect } from 'react';
import {
  Container,
  Textarea,
  Button,
  Title,
  Text,
  Loader,
  Box,
  Paper,
  Group,
  ScrollArea,
  Stack,
  useMantineTheme,
} from '@mantine/core';

interface Message {
  role: 'user' | 'ai';
  content: string;
}

export default function Home() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const theme = useMantineTheme();

  const ask = async () => {
    if (!input.trim()) return;
    const userMessage = { role: 'user' as const, content: input };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);
    setInput('');

    try {
      const res = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: input, hybrid: true, filter: '' }),
      });
      const data = await res.json();
      const aiMessage = { role: 'ai' as const, content: data.answer };
      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: 'ai', content: '❌ Error fetching response. Please try again.' },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      ask();
    }
  };

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <Container size="sm" py="xl">
      <Title order={2} mb="md">
        🛡️ System Safety AI Assistant
      </Title>

      <ScrollArea h={400} type="always" offsetScrollbars>
        <Stack spacing="sm">
          {messages.map((msg, idx) => (
            <Group key={idx} position={msg.role === 'user' ? 'right' : 'left'}>
              <Paper
                shadow="xs"
                p="sm"
                radius="md"
                withBorder
                bg={msg.role === 'user' ? theme.colors.blue[0] : theme.colors.gray[0]}
              >
                <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>
                  {msg.content}
                </Text>
              </Paper>
            </Group>
          ))}
          <div ref={scrollRef} />
        </Stack>
      </ScrollArea>

      <Box mt="md">
        <Textarea
          value={input}
          onChange={(e) => setInput(e.currentTarget.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your question and press Enter..."
          minRows={3}
          autosize
        />
        <Group mt="sm" position="right">
          <Button onClick={ask} disabled={loading}>
            {loading ? <Loader size="xs" /> : 'Submit'}
          </Button>
        </Group>
      </Box>
    </Container>
  );
}
