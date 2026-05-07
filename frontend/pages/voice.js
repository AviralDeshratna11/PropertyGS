import VoiceAssistant from '../components/VoiceAssistant'

export default function VoicePage(){
  return (
    <div>
      <h1 className="text-2xl font-semibold mb-4">Voice Assistant Demo</h1>
      <VoiceAssistant propertyId={1} />
    </div>
  )
}
